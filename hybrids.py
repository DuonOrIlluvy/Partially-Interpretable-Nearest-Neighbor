"""
This file contains implementation of Frost's hybrid decision list and
hybrid nearest neighbor (compatible with sklearn),
along with some helper function and dataframe functions 
(such as creating dataframe fit for pareto frontier plot).
"""

import pandas as pd 
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from typing import Iterator, cast
import numpy.typing as npt
from param import Boolean
from tqdm import tqdm
import multiprocessing as mp
from typing import Literal
from sklearn.base import BaseEstimator, ClassifierMixin


def max_pareto_front(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Remove any point in given df that is not pareto efficient with respect to given columns
    """
    if df.empty:
        return df
    pf_list = []
    sorted_df = df.sort_values(by=[col1, col2], ascending=[False, False])
    max_col2 = float("-inf")
    for _, row in sorted_df.iterrows():
        if row[col2] > max_col2:
            max_col2 = row[col2]
            pf_list.append(row)
    return pd.DataFrame(pf_list)



def categorize_numericals(df_train: pd.DataFrame, Nlevel: int = 5, df_test: pd.DataFrame | None = None
                          ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns new dataframe with each numerical column replaced with Nlevel quantile categorical columns,
    values specifies if the original value belongs to corresponding upper quantiles.
    The implementation is almost the same as HyRS repo's code_continuous function.

    If test dataframe is also specified, their columns will also be categorized based on the training set's quantiles.
    """
    numericals = [col for col in df_train.columns if df_train[col].dtype in [int, float] and (~df_train[col].isin([0, 1])).any()]
    
    new_cols_train = []
    new_cols_test = []

    for col in numericals:
        for q in range(1, Nlevel):
            threshold = df_train[col].quantile(q / Nlevel)
            new_col_train = (df_train[col] > threshold).astype(int)
            new_col_train.name = f"{col}>={threshold}"
            new_cols_train.append(new_col_train)

            if df_test is not None:
                new_col_test = (df_test[col] > threshold).astype(int)
                new_col_test.name = f"{col}>={threshold}"
                new_cols_test.append(new_col_test)


    new_df_train = pd.concat([df_train.drop(columns=numericals)] + new_cols_train, axis=1)
    if df_test is not None:
        new_df_test = pd.concat([df_test.drop(columns=numericals)] + new_cols_test, axis=1)
        return new_df_train, new_df_test

    return new_df_train


def covers(rules: list[frozenset[int]], X: np.ndarray) -> npt.NDArray[np.int64]:
    """ Returns for each row in X if it is covered by at least one rule in the rules. """
    rules_np = [list(rule) for rule in rules]
    X_covered = np.any([np.all(X[:, rule], axis=1) for rule in rules_np], axis=0)
    # print(f"rule sample: {rules_np[0]}") if rules_np.size and X_items else None
    # print(f"X sample: {X_items[0]}") if rules_np.size and X_items else None
    return np.array(X_covered)


def support(rules: list[frozenset[int]], X: np.ndarray) -> int:
    """ Returns the amount of row in X that is covered by the rules. """
    return covers(rules, X).sum()


def cover_index(rules: list[frozenset[int]], x: np.ndarray) -> int:
    """Returns first index of given ruleset that covers given instance"""
    x_set = set(x)
    for i, rule in enumerate(rules):
        if x_set >= rule:
            return i
    
    return -1


def calculate_posrule_covers(rule_data):
    """
    Calculate covers and misprediction counts for positive rule
    """
    rulelen, uplus, X_set, Y_bool = rule_data
    rule_cover = np.zeros(rulelen)
    rule_miscover = np.zeros(rulelen)
    for i, rule in enumerate(uplus):
        covered = (X_set >= rule)
        miscovered = covered & (~Y_bool)
        rule_cover[i] = covered.sum()
        rule_miscover[i] = miscovered.sum()
    return np.array((rule_cover, rule_miscover))


def calculate_negrule_covers(rule_data):
    """
    Calculate covers and misprediction counts for negative rule
    """
    rulelen, upneg, X_set, Y_bool = rule_data
    rule_cover = np.zeros(rulelen)
    rule_miscover = np.zeros(rulelen)
    for i, rule in enumerate(upneg):
        covered = (X_set >= rule)
        miscovered = covered & Y_bool
        rule_cover[i] = covered.sum()
        rule_miscover[i] = miscovered.sum()
    return np.array((rule_cover, rule_miscover))


def calculate_posrule_mask(rule_data):
    """
    Calculate covers and misprediction counts for positive rule with bitpacked data
    """
    rulelen, uplus, X_packed, notY_packed = rule_data
    covers = np.zeros(rulelen, dtype=int)
    miscovers = np.zeros(rulelen, dtype=int)
    for i, cols in tqdm(enumerate(uplus), total=rulelen):
        cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
        covers[i] = np.bitwise_count(cov).sum()
        miscovers[i] = np.bitwise_count(np.bitwise_and(cov, notY_packed)).sum()
    return np.array((covers, miscovers))


def calculate_negrule_mask(rule_data):
    """
    Calculate covers and misprediction counts for negative rule with bitpacked data
    """
    rulelen, upneg, X_packed, Y_packed = rule_data
    covers = np.zeros(rulelen, dtype=int)
    miscovers = np.zeros(rulelen, dtype=int)
    for i, cols in tqdm(enumerate(upneg), total=rulelen):
        cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
        covers[i] = np.bitwise_count(cov).sum()
        miscovers[i] = np.bitwise_count(np.bitwise_and(cov, Y_packed)).sum()
    return np.array((covers, miscovers))


class hyb_rules_frost():
    """
    Implementation of Hybrid decision list training algorithm
    from paper "Partially Interpretable Models with
    Guarantees on Coverage and Accuracy".
    """
    def __init__(self):
        self.rules: list[tuple[frozenset[int], int]] = []
        self.s = 0.0
        self.l = 0.0

    @property
    def conditions(self) -> int:
        """Sum of conditions present in fitted rule set"""
        return sum([len(rule) for rule in self.rules])
    
    
    def predict(self, X: pd.DataFrame | np.ndarray, Yb: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """
        Make a prediction using fitted parameters.

        Parameters
        -----------
        X : pandas DataFrame
            Dataframe with only -1 and 1, 0 and 1, or True and False, containing datas needed to predict.

        Yb : pandas DataFrame or numpy array
            Containing values predicted by a black-box model.

        Returns
        -----------
        Numpy array containing predicted values (yhat), and list of indices of datapoints covered by fitted ruleset instead of black-box model.
        """
        X_np = X if isinstance(X, np.ndarray) else X.to_numpy()
        X_set = [set(x.nonzero()[0]) for x in (X_np == 1)]
        Yhat = Yb.copy()
        covered_indices = []

        for i, x_set in enumerate(X_set):
            for rule, pred in self.rules:
                if x_set >= rule:
                    Yhat[i] = pred
                    covered_indices.append(i)  
                    break
        
        return Yhat, covered_indices
    
    def fit(
            self, binary_df: pd.DataFrame, Y: pd.DataFrame | np.ndarray,
            l: float, s: float, maxlen: int | None = None
            ) -> None:
        """
        Train the model using Algorithm 1 from 
        "Partially Interpretable Models with Guarantees on Coverage and Accuracy".

        Parameters
        -----------
        binary_df : pandas DataFrame
            Dataframe with only -1 and 1, 0 and 1, or True and False, containing datas needed to predict.
        
        Y : pandas DataFrame or numpy array
            Containing values to try to predict.
        
        l: float
            Fraction covered to be considered large set.
            Any rule that does not cover more than l fraction of the dataframe is ignored.

        s: float
            Fraction of examples to collect before the algorithm stops.

        maxlen : int or None (default: None)
            Integer specifying maximum amount of condition in a single rule. 
            Setting it None will make it unbounded
        """
        self.s = s
        self.l = l
        # Prepare datasets
        binary_df_01 = binary_df.astype(int).replace(-1, 0)
        X = binary_df_01.to_numpy().astype(bool)
        X_packed = np.packbits(X, axis=0)
        N = len(binary_df_01)
        Y_np = cast(np.ndarray, Y).copy()
        Y_np[Y_np == 0] = -1
        Y_bool = (Y_np == 1)
        Y_packed = np.packbits(Y_bool)
        notY_packed = np.packbits(~Y_bool)
        pindex = (Y_np == 1).nonzero()[0]
        nindex = (Y_np == -1).nonzero()[0]
        p_fraction = len(pindex)/N
        n_fraction = len(nindex)/N

        
        print("Generating initial rulesets...")
        # List of potential rules to predict positive
        upsilon_plus = []
        if p_fraction > 0:
            upsilon_plus = fpgrowth(
                binary_df_01.iloc[pindex].astype(bool),
                min_support=min(1, l/p_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        # List of potential rules to predict negative
        upsilon_minus = []
        if n_fraction > 0:
            upsilon_minus = fpgrowth(
                binary_df_01.iloc[nindex].astype(bool),
                min_support=min(1, l/n_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        upsilon = np.hstack([upsilon_plus, upsilon_minus])
        prulelen = len(upsilon_plus)
        nrulelen = len(upsilon_minus)
        if upsilon.size == 0:
            print("Did not generate any rule from fpgrowth!")
            return
        print(f"Generated {prulelen} positive rules and {nrulelen} negative rules")
        rule_count = len(upsilon)

        # covers[i,j] = True if point j obeys rule i else False (packbits to try to reduce RAM usage)
        covers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        # miscovers[i,j] = True if rule i predicts point j wrong else False
        miscovers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        print("Calculating covers for each rules...")
        for i, rule in tqdm(enumerate(upsilon_plus), total=prulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[i] = cov
            miscovers[i] = np.bitwise_and(cov, notY_packed)
        for i, rule in tqdm(enumerate(upsilon_minus), total=nrulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[prulelen+i] = cov
            miscovers[prulelen+i] = np.bitwise_and(cov, Y_packed)
        
        t = 1
        remaining = np.packbits(np.full(N, True, dtype=bool))
        coverage_so_far = 0
        self.rules = []

        while coverage_so_far < s:
            print(f"Iteration {t}: ")
            print(f"coverage_so_far: {coverage_so_far}")
            # total_counts[i] = amount of points that obeys rule i
            total_counts = np.bitwise_count(np.bitwise_and(covers, remaining)).sum(axis=1)
            # total_miscounts[i] = amount of points that gets mispredicted by rule i
            total_miscounts = np.bitwise_count(np.bitwise_and(miscovers, remaining)).sum(axis=1)
            # Don't consider rules that covers too little points
            valid_rules = total_counts >= (l * N)
            if not np.any(valid_rules):
                print("No more rule to add")
                break  
            # ratio of mispredicted / covered on remaining datapoints
            ratio = np.full_like(total_counts, np.inf, dtype=float)
            ratio[valid_rules] = total_miscounts[valid_rules] / total_counts[valid_rules]
            # Best performing rule
            r_index = int(np.nanargmin(ratio))
            coverage_so_far += total_counts[r_index] / N
            # remove instance from remaining that is covered by new rule
            covered_indices = covers[r_index]
            remaining = np.bitwise_and(remaining, np.bitwise_not(covered_indices))
            # Add the best performing rule
            rule_truth = 1 if r_index < prulelen else 0
            self.rules.append((upsilon[r_index], rule_truth))
            print(f"added rule: {upsilon[r_index]} -> {rule_truth}")

            t += 1
        else:
            print(f"Target coverage ({coverage_so_far} > {s}) reached!")

    def fit_yield(
            self, binary_df: pd.DataFrame, Y: pd.DataFrame | np.ndarray,
            l: float, s_list: list[float], maxlen: int | None = None
            ) -> Iterator[None]:
        """
        Fit that takes a list of s instead, 
        reusing the generated candidate rules and coverage counts for each s.
        This returns an iterator where each iteration will fit the model with next s.

        Parameters
        -----------
        binary_df : pandas DataFrame
            Dataframe with only -1 and 1, 0 and 1, or True and False, containing datas needed to predict.
        
        Y : pandas DataFrame or numpy array
            Containing values to try to predict.
        
        l: float
            Fraction covered to be considered large set.
            Any rule that does not cover more than l fraction of the dataframe is ignored.

        s_list: list[float]
            List of s, has to be increasing.
            The function will yield at each point where the coverage exceeds current s.

        maxlen : int or None (default: None)
            Integer specifying maximum amount of condition in a single rule. 
            Setting it None will make it unbounded
        """
        # Prepare datasets
        binary_df_01 = binary_df.astype(int).replace(-1, 0)
        X = binary_df_01.to_numpy().astype(bool)
        X_packed = np.packbits(X, axis=0)
        N = len(binary_df_01)
        Y_np = cast(np.ndarray, Y).copy()
        Y_np[Y_np == 0] = -1
        Y_bool = (Y_np == 1)
        Y_packed = np.packbits(Y_bool)
        notY_packed = np.packbits(~Y_bool)
        pindex = (Y_np == 1).nonzero()[0]
        nindex = (Y_np == -1).nonzero()[0]
        p_fraction = len(pindex)/N
        n_fraction = len(nindex)/N

        
        print("Generating initial rulesets...")
        # List of potential rules to predict positive
        upsilon_plus = []
        if p_fraction > 0:
            upsilon_plus = fpgrowth(
                binary_df_01.iloc[pindex].astype(bool),
                min_support=min(1, l/p_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        # List of potential rules to predict negative
        upsilon_minus = []
        if n_fraction > 0:
            upsilon_minus = fpgrowth(
                binary_df_01.iloc[nindex].astype(bool),
                min_support=min(1, l/n_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        upsilon = np.hstack([upsilon_plus, upsilon_minus])
        prulelen = len(upsilon_plus)
        nrulelen = len(upsilon_minus)
        if upsilon.size == 0:
            print("Did not generate any rule from fpgrowth!")
            return
        print(f"Generated {prulelen} positive rules and {nrulelen} negative rules")

        # covers[i,j] = True if point j obeys rule i else False (packbits to try to reduce RAM usage)
        covers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        # miscovers[i,j] = True if rule i predicts point j wrong else False
        miscovers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        print("Calculating covers for each rules...")
        for i, rule in tqdm(enumerate(upsilon_plus), total=prulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[i] = cov
            miscovers[i] = np.bitwise_and(cov, notY_packed)
        for i, rule in tqdm(enumerate(upsilon_minus), total=nrulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[prulelen+i] = cov
            miscovers[prulelen+i] = np.bitwise_and(cov, Y_packed)
        
        t = 1
        remaining = np.packbits(np.full(N, True, dtype=bool))
        coverage_so_far = 0
        self.rules = []
        
        self.l = l

        for s in s_list:
            print(f"Continuing with s: {s}")
            self.s = s
            while coverage_so_far < s:
                print(f"Iteration {t}: ")
                print(f"coverage_so_far: {coverage_so_far}")
                # total_counts[i] = amount of points that obeys rule i
                total_counts = np.bitwise_count(np.bitwise_and(covers, remaining)).sum(axis=1)
                # total_miscounts[i] = amount of points that gets mispredicted by rule i
                total_miscounts = np.bitwise_count(np.bitwise_and(miscovers, remaining)).sum(axis=1)
                # Don't consider rules that covers too little points
                valid_rules = total_counts >= (l * N)
                if not np.any(valid_rules):
                    print("No more rule to add")
                    return
                # ratio of mispredicted / covered on remaining datapoints
                ratio = np.full_like(total_counts, np.inf, dtype=float)
                ratio[valid_rules] = total_miscounts[valid_rules] / total_counts[valid_rules]
                # Best performing rule
                r_index = int(np.nanargmin(ratio))
                coverage_so_far += total_counts[r_index] / N
                # remove instance from remaining that is covered by new rule
                covered_indices = covers[r_index]
                remaining = np.bitwise_and(remaining, np.bitwise_not(covered_indices))
                # Add the best performing rule
                rule_truth = 1 if r_index < prulelen else 0
                self.rules.append((upsilon[r_index], rule_truth))
                print(f"added rule: {upsilon[r_index]} -> {rule_truth}")

                t += 1
            else:
                print(f"Target coverage ({coverage_so_far} > {s}) reached!")
                yield

    def fit_yield_gpu(
            self, binary_df: pd.DataFrame, Y: pd.DataFrame | np.ndarray,
            l: float, s_list: list[float], maxlen: int | None = None
            ) -> Iterator[None]:
        """
        fit_yield that uses CuPy for GPU acceleration.

        Parameters
        -----------
        binary_df : pandas DataFrame
            Dataframe with only -1 and 1, 0 and 1, or True and False, containing datas needed to predict.
        
        Y : pandas DataFrame or numpy array
            Containing values to try to predict.
        
        l: float
            Fraction covered to be considered large set.
            Any rule that does not cover more than l fraction of the dataframe is ignored.

        s_list: list[float]
            List of s, has to be increasing.
            The function will yield at each point where the coverage exceeds current s.

        maxlen : int or None (default: None)
            Integer specifying maximum amount of condition in a single rule. 
            Setting it None will make it unbounded
        """
        import cupy as cp

        # Prepare datasets
        binary_df_01 = binary_df.astype(int).replace(-1, 0)
        X = binary_df_01.to_numpy().astype(bool)
        X_packed = np.packbits(X, axis=0)
        N = len(binary_df_01)
        Y_np = cast(np.ndarray, Y).copy()
        Y_np[Y_np == 0] = -1
        Y_bool = (Y_np == 1)
        Y_packed = np.packbits(Y_bool)
        notY_packed = np.packbits(~Y_bool)
        pindex = (Y_np == 1).nonzero()[0]
        nindex = (Y_np == -1).nonzero()[0]
        p_fraction = len(pindex)/N
        n_fraction = len(nindex)/N
        print(n_fraction)
        print(p_fraction)

        
        print("Generating initial rulesets...")
        # List of potential rules to predict positive
        upsilon_plus = []
        if p_fraction > 0:
            upsilon_plus = fpgrowth(
                binary_df_01.iloc[pindex].astype(bool),
                min_support=min(1, l/p_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        # List of potential rules to predict negative
        upsilon_minus = []
        if n_fraction > 0:
            upsilon_minus = fpgrowth(
                binary_df_01.iloc[nindex].astype(bool),
                min_support=min(1, l/n_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        upsilon = np.hstack([upsilon_plus, upsilon_minus])
        prulelen = len(upsilon_plus)
        nrulelen = len(upsilon_minus)
        if upsilon.size == 0:
            print("Did not generate any rule from fpgrowth!")
            return
        print(f"Generated {prulelen} positive rules and {nrulelen} negative rules")

        # covers[i,j] = True if point j obeys rule i else False (packbits to try to reduce RAM usage)
        covers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        # miscovers[i,j] = True if rule i predicts point j wrong else False
        miscovers = np.zeros((len(upsilon), int(np.ceil(N / 8))), dtype=np.uint8)
        print("Calculating covers for each rules...")
        for i, rule in tqdm(enumerate(upsilon_plus), total=prulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[i] = cov
            miscovers[i] = np.bitwise_and(cov, notY_packed)
        for i, rule in tqdm(enumerate(upsilon_minus), total=nrulelen):
            cols = list(rule)
            cov = np.bitwise_and.reduce(X_packed[:, cols], axis=1)
            covers[prulelen+i] = cov
            miscovers[prulelen+i] = np.bitwise_and(cov, Y_packed)
        
        t = 1
        remaining = np.packbits(np.full(N, True, dtype=bool))
        coverage_so_far = 0
        self.rules = []
        
        self.l = l
        # send to gpu
        covers = cp.asarray(covers)
        miscovers = cp.asarray(miscovers)
        remaining = cp.asarray(remaining)
        X_packed = cp.asarray(X_packed)

        for s in s_list:
            print(f"Continuing with s: {s}")
            self.s = s
            while coverage_so_far < s:
                print(f"Iteration {t}: ")
                print(f"coverage_so_far: {coverage_so_far}")
                # total_counts[i] = amount of points that obeys rule i
                total_counts = cp.bitwise_count(cp.bitwise_and(covers, remaining)).sum(axis=1)
                # total_miscounts[i] = amount of points that gets mispredicted by rule i
                total_miscounts = cp.bitwise_count(cp.bitwise_and(miscovers, remaining)).sum(axis=1)
                # Don't consider rules that covers too little points
                valid_rules = total_counts >= (l * N)
                if not cp.any(valid_rules):
                    print("No more rule to add")
                    return
                # ratio of mispredicted / covered on remaining datapoints
                ratio = cp.full_like(total_counts, cp.inf, dtype=float)
                ratio[valid_rules] = total_miscounts[valid_rules] / total_counts[valid_rules]
                # Best performing rule
                r_index = int(cp.nanargmin(ratio))
                coverage_so_far += total_counts[r_index] / N
                # remove instance from remaining that is covered by new rule
                covered_indices = covers[r_index]
                remaining = cp.bitwise_and(remaining, cp.bitwise_not(covered_indices))
                # Add the best performing rule
                rule_truth = 1 if r_index < prulelen else 0
                self.rules.append((upsilon[r_index], rule_truth))
                print(f"added rule: {upsilon[r_index]} -> {rule_truth}")

                t += 1
            else:
                print(f"Target coverage ({coverage_so_far} > {s}) reached!")
                yield
        
            
    def fit_yield_lowmem(
            self, binary_df: pd.DataFrame, Y: pd.DataFrame | np.ndarray,
            l: float, s_list: list[float], maxlen: int | None = None
            ) -> Iterator[None]:
        """
        Alternate fit function that does not create array with size of (canditate rule count * len(binary_df)).
        It runs slower but may be the only way to fit when the initial fpgrowth creates like 10000000 candidate rules
        for very large dataframe with low l.
        It will take a list of s instead, reusing the generated candidate rules and coverage counts for each s.

        Parameters
        -----------
        binary_df : pandas DataFrame
            Dataframe with only -1 and 1, 0 and 1, or True and False, containing datas needed to predict.
        
        Y : pandas DataFrame or numpy array
            Containing values to try to predict.
        
        l: float
            Fraction covered to be considered large set.
            Any rule that does not cover more than l fraction of the dataframe is ignored.

        s_list: list[float]
            List of s, has to be increasing.
            The function will yield at each point where the coverage exceeds current s.

        maxlen : int or None (default: None)
            Integer specifying maximum amount of condition in a single rule. 
            Setting it None will make it unbounded
        """
        # Prepare datasets
        binary_df_01 = binary_df.astype(int).replace(-1, 0)
        X = binary_df_01.to_numpy().astype(bool)
        X_packed = np.packbits(X, axis=0)
        N = len(binary_df_01)
        Y_np = cast(np.ndarray, Y).copy()
        Y_np[Y_np == 0] = -1
        Y_bool = (Y_np == 1)
        Y_packed = np.packbits(Y_bool)
        notY_packed = np.packbits(~Y_bool)
        pindex = (Y_np == 1).nonzero()[0]
        nindex = (Y_np == -1).nonzero()[0]
        p_fraction = len(pindex)/N
        n_fraction = len(nindex)/N

        print("Generating initial rulesets...")
        # List of potential rules to predict positive
        upsilon_plus = []
        if p_fraction > 0:
            upsilon_plus = fpgrowth(
                binary_df_01.iloc[pindex].astype(bool),
                min_support=min(1, l/p_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        # List of potential rules to predict negative
        upsilon_minus = []
        if n_fraction > 0:
            upsilon_minus = fpgrowth(
                binary_df_01.iloc[nindex].astype(bool),
                min_support=min(1, l/n_fraction),
                max_len=maxlen
            )["itemsets"].to_numpy()  
        upsilon = np.hstack([upsilon_plus, upsilon_minus])
        if upsilon.size == 0:
            print("Did not generate any rule from fpgrowth!")
            return
        upsilon_minus_lis = [np.array(list(rule)) for rule in upsilon_minus]
        upsilon_plus_lis = [np.array(list(rule)) for rule in upsilon_plus]
        upsilon_lis = upsilon_plus_lis + upsilon_minus_lis
        prulelen = len(upsilon_plus)
        nrulelen = len(upsilon_minus)
        print(f"Generated {prulelen} positive rules and {nrulelen} negative rules")

        def list_split(lis: list, count: int):
            """
            Splits list into 'count' chunks to be fed independently into processor pool
            """
            lenlis = len(lis)
            chunk = lenlis//count
            i = 0
            for i in range(count):
                yield lis[i*chunk:(i+1)*chunk]
            if (i+1)*chunk < lenlis - 1:
                yield lis[(i+1)*chunk:]
        
        print("Calculating initial cover counts...")
        posrule_data = [(len(uplus), uplus, X_packed, notY_packed) for uplus in list_split(upsilon_plus_lis, mp.cpu_count())]
        negrule_data = [(len(upneg), upneg, X_packed, Y_packed) for upneg in list_split(upsilon_minus_lis, mp.cpu_count())]
        with mp.Pool() as pool:
            results = np.hstack(
                pool.map(calculate_posrule_mask, posrule_data) +
                pool.map(calculate_negrule_mask, negrule_data)
                )
        # total_counts[i] = amount of points that obeys rule i
        total_counts = results[0]
        # total_miscounts[i] = amount of points that gets mispredicted by rule i
        total_miscounts = results[1]
        
        t = 1
        remaining = np.packbits(np.full(N, True, dtype=bool))
        coverage_so_far = 0
        self.rules = []
        self.l = l

        for s in s_list:
            print(f"Continuing with s: {s}")
            self.s = s
            while coverage_so_far < s:
                print(f"Iteration {t}: ")
                print(f"coverage_so_far: {coverage_so_far}")
                # Don't consider rules that covers too little points
                valid_rules = total_counts >= (l * N)
                if not np.any(valid_rules):
                    print("No more rule to add")
                    return
                # ratio of mispredicted / covered on remaining datapoints
                ratio = np.full_like(total_counts, np.inf, dtype=float)
                ratio[valid_rules] = total_miscounts[valid_rules] / total_counts[valid_rules]
                # Best performing rule
                r_index = int(np.nanargmin(ratio))
                coverage_so_far += total_counts[r_index] / N
                # remove instance from remaining that is covered by new rule
                covered_mask = np.bitwise_and.reduce(X_packed[:, upsilon_lis[r_index]], axis=1)
                covered_diff = np.bitwise_and(remaining, covered_mask)
                remaining = np.bitwise_and(remaining, np.bitwise_not(covered_mask))
                # Add the best performing rule
                rule_truth = 1 if r_index < prulelen else 0
                self.rules.append((upsilon[r_index], rule_truth))
                print(f"added rule: {upsilon[r_index]} -> {rule_truth}")
                
                print("Recalculating cover counts...")
                X_packed_filtered = np.bitwise_and(X_packed, covered_diff[:, None])
                Y_packed_filtered = np.bitwise_and(Y_packed, covered_diff)
                notY_packed_filtered = np.bitwise_and(notY_packed, covered_diff)
                posrule_data = [(len(uplus), uplus, X_packed_filtered, notY_packed_filtered) for uplus in list_split(upsilon_plus_lis, mp.cpu_count())]
                negrule_data = [(len(upneg), upneg, X_packed_filtered, Y_packed_filtered) for upneg in list_split(upsilon_minus_lis, mp.cpu_count())]
                with mp.Pool() as pool:
                    results = np.hstack(
                        pool.map(calculate_posrule_mask, posrule_data) +
                        pool.map(calculate_negrule_mask, negrule_data)
                        )

                total_counts -= results[0]
                total_miscounts -= results[1]
                t += 1
            else:
                print(f"Target coverage ({coverage_so_far} > {s}) reached!")
                yield
            

class hyb_lin:
    """
    Implementation of Hybrid linear model from the paper
    "Hybrid Predictive Models: When an Interpretable Model
    Collaborates with a Black-box Model".

    Instead of Algorithm 2, it uses CYXPY to minimize the loss function (12) in the paper directly
    """
    
    def __init__(self):
        self.alpha_1 = 0.0
        self.alpha_2 = 0.0
        self.theta_plus = 0.0
        self.theta_minus = 0.0
        self.w: npt.NDArray | None = None
    
    @property
    def interpretability(self):
        """Amount of zero weights i.e. amount of unused features"""
        return np.count_nonzero(self.w == 0)
    
    @property
    def used_features(self):
        """Amount of non-zero weights i.e. amount of used features"""
        return np.count_nonzero(self.w != 0)
    
    def fit(
            self, df: pd.DataFrame, Y: pd.DataFrame | np.ndarray, Yb: pd.DataFrame | np.ndarray, 
            alpha_1: float, alpha_2: float, zero_thrs: float = 1e-10, *, verbose = False
            ) -> str:
        """
        Train the model using CYXPY to minimize the (12) loss function from the paper.

        Parameters
        -----------
        df : pandas DataFrame
            Dataframe containing datas needed to predict.
        
        Y : pandas DataFrame or numpy array
            Containing values to try to predict.
        
        Yb : pandas DataFrame or numpy array
            Containing values predicted by a black-box model.
        
        alpha_1 : float
            Value that multiplies on lasso regularization inside (12). Higher values will make the model have
            more zero weights, meaning it will use less features to predict, making the model more interpretable.
        
        alpha_2 : float
            Value that multiplies on difference between theta plus and minus inside (12). 
            Higher value will lessen this difference, which will make prediction by the linear model instead of
            the black-box model more often, increasing its transparancy. Must be non-negative.
        
        zero_thrs : float (default : 1e-10)
            Weights with absolute value below this value will be rounded down to zero during training.

        verbose : bool (default: False)
            Enable CVXPY's verbose solving.

        Returns
        -------
        Problem.status after solving. If it says it could not find optimum, then the fitting was unsuccessful.
        """
        import cvxpy as cp

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        X = df.to_numpy()
        (N, d) = X.shape

        Yb_bool = Yb.astype(bool)
        # Because Y is 0 or 1 while paper expects -1 or 1
        Y_pm1 = np.where(Y > 0, 1, -1)
        # Define variables and parameters
        w = cp.Variable(d)
        theta_plus = cp.Variable(1)
        theta_minus = cp.Variable(1)

        # masking X and Y in terms of Yb (I^+_b and I^-_b)
        Xp = X[Yb_bool]
        Yp = Y_pm1[Yb_bool]

        Xm = X[~Yb_bool]
        Ym = Y_pm1[~Yb_bool]

        # logistic loss
        loss_Ib_plus = cp.sum(cp.logistic(-cp.multiply(Yp, Xp @ w - theta_minus)))
        loss_Ib_minus = cp.sum(cp.logistic(-cp.multiply(Ym, Xm @ w - theta_plus)))
        loss = (loss_Ib_plus + loss_Ib_minus)/N

        # regularization
        reg = alpha_1*cp.norm1(w) + alpha_2*(theta_plus - theta_minus)

        # objective / loss function
        obj = cp.Minimize(loss + reg)

        constraints = [theta_plus >= theta_minus]

        # solve the CVXPY problem
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=verbose)
        if w.value is not None:
            self.w = np.where(np.abs(w.value) < zero_thrs, 0, w.value)
            self.theta_plus = theta_plus.value
            self.theta_minus = theta_minus.value
        else:
            self.w = None
        
        return prob.status
    
    def predict(self, X: pd.DataFrame | np.ndarray, Yb: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[int]]:
        """
        Make a prediction using fitted parameters.

        Parameters
        -----------
        X : pandas DataFrame
            Dataframe containing datas needed to predict.

        Yb : pandas DataFrame or numpy array
            Containing values predicted by a black-box model.

        Returns
        -----------
        Numpy array containing predicted values (yhat), and list of indices of datapoints covered by fitted linear model instead of black-box model.
        """
        X_np = X if isinstance(X, np.ndarray) else X.to_numpy()
        Yhat = np.zeros_like(Yb)
        covered = []
        for (i, datapoint) in enumerate(X_np):
            value = np.inner(self.w, datapoint)
            if value > self.theta_plus:  # predict positive by simple linear model
                Yhat[i] = 1
                covered.append(i)
            elif value < self.theta_minus:  # predict negative by simple linear model
                Yhat[i] = 0
                covered.append(i)
            else:  # predict using black-box model
                Yhat[i] = Yb[i]
        
        return Yhat, covered
    


class hyb_KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """
    KNeighborsClassifier with a functionality to defer to a black-box model's output when
    the vote proportion is too close to 0.5.
    It will only work for binary classification.

    Added Parameter
    ----------
    defer_threshold : float, default=0.2
        When |vote proportion - 0.5| < defer_threshold, the prediction will come from an output from
        provided black-box model's output.

    gpu: bool, default=False
        Whether to activate GPU acceleration via cuml and cupy.

    probability: bool, default=False
        If True, use the PNeighborsClassifier to do predict_proba()
        instead of KNeighborsClassifier from sklearn or cuml.

    Everything else is the same as KNeighborsClassifier.
    """
    def __init__(
        self,
        n_neighbors=5,
        defer_threshold=0.2,
        gpu=False,
        *,
        weights="uniform",
        probability=False,
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        output_type: str | None = None,
        verbose: int | Boolean = False,
        algo_params: dict | None = None,
        # **kwargs
    ):
        self.gpu = gpu
        self.probability = probability
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.output_type = output_type
        self.verbose = verbose
        self.algo_params = algo_params
        if probability:
            self.knn = PNeighborsClassifier(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                weights=weights,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
                gpu=gpu
            )
        elif gpu:
            from cuml.neighbors import KNeighborsClassifier as cuKNN
            self.knn = cuKNN(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                weights=weights,
                metric=metric,
                output_type=output_type,
                verbose=verbose
            )
        else:
            from sklearn.neighbors import KNeighborsClassifier as skKNN
            self.knn = skKNN(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                weights=weights,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )
        self.defer_threshold = defer_threshold
        self._estimator_type = "classifier"
        self.verbose = verbose
    
    def set_params(self, **params):
        # handle added parameter parameter first
        if "defer_threshold" in params:
            self.defer_threshold = params.pop("defer_threshold")

        # copy and remove keys unsupported by the underlying KNN
        params = params.copy()
        if self.gpu:
            # remove sklearn-only params when using cuML
            params.pop("leaf_size", None)
            params.pop("metric_params", None)
            params.pop("p", None)
            params.pop("n_jobs", None)
        else:
            # remove cuML-only params when using sklearn
            params.pop("verbose", None)
            params.pop("output_type", None)

        self.knn.set_params(**params)
        return self
    
    def get_params(self, deep: bool = True) -> dict:
        # all the possible params params 
        params = {
            'n_neighbors': getattr(self, 'n_neighbors', 5),
            'defer_threshold': getattr(self, 'defer_threshold', 0.2),
            'gpu': getattr(self, 'gpu', False),
            'weights': getattr(self, 'weights', "uniform"),
            'algorithm': getattr(self, 'algorithm', "auto"),
            'leaf_size': getattr(self, 'leaf_size', 20),
            'p': getattr(self, 'p', 2),
            'metric': getattr(self, 'metric', "minkowski"),
            'metric_params': getattr(self, 'metric_params', None),
            'n_jobs': getattr(self, 'n_jobs', None),
            'output_type': getattr(self, 'output_type', None),
            'verbose': getattr(self, 'verbose', False),
            'algo_params': getattr(self, 'algo_params', None),
        }

        # Get params from self.knn, add any params missing so that GridSearchCV does not complain
        try:
            knn_params = self.knn.get_params(deep=deep)
        except:
            knn_params = {}

        for k, v in knn_params.items():
            if k not in params:
                params[k] = v

        return params

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Training data. Last column is assumed to be an output from a black-box model.
            The rest of columns are thrown into KNeighborsClassifier's .fit function.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        self.classes_, y = np.unique(y, return_inverse=True)
        if X.shape[1] <= 1:
            return self
        return self.knn.fit(X[:, :-1], y)
    
    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Test sample. Last column is assumed to be an output from a black-box model.
            That output will be used when this hybrid model decides to defer.
            The rest of columns are thrown into KNeighborsClassifier's .predict function.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        if X.shape[1] <= 1:
            return X.astype(int)
        prob = self.knn.predict_proba(X[:, :-1])[:, 1]
        Yhat = (prob >= 0.5)
        hyb_yhat = np.copy(Yhat)
        mask = np.abs(prob - 0.5) < self.defer_threshold
        hyb_yhat[mask] = X[:, -1][mask]
        return hyb_yhat
    
    def score_detail(self, X, y):
        """Predict the class labels for the provided data, and
        return a prediction detail.

        Parameters
        ----------
        X : numpy.ndarray
            Test sample. Last column is assumed to be an output from a black-box model.
            That output will be used when this hybrid model decides to defer.
            The rest of columns are thrown into KNeighborsClassifier's .predict function.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        Returns
        -------
        dictionary:
            {
                'offload_thrs': model's defer_threshold,
                'transparency': fraction test datapoint covered by knn instead of black-box,
                'accuracy': mean accuracy of the hybrid model's prediction
            }
        """
        if X.shape[1] <= 1:
            # Only black-box column?
            return {
                'offload_thrs': self.defer_threshold,
                'transparency': 0.0,
                'accuracy': np.mean(X.astype(int) == y)
            }
        test_num = len(y)
        Ybtest = X[:, -1]

        prob = self.knn.predict_proba(X[:, :-1])[:, 1]
        Yhat = (prob >= 0.5)
        hyb_yhat = np.copy(Yhat)
        mask = np.abs(prob - 0.5) < self.defer_threshold
        hyb_yhat[mask] = Ybtest[mask]

        covered_amount = test_num - np.sum(mask)
        correct_amount = np.count_nonzero(hyb_yhat == y)
        transparency = covered_amount / test_num
        accuracy = correct_amount / test_num
        return {
            'offload_thrs': self.defer_threshold,
            'transparency': transparency,
            'accuracy': accuracy
        }

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : numpy.ndarray
            Test sample. Last column is assumed to be an output from a black-box model.
            That output will be used when this hybrid model decides to defer.
            The rest of columns are thrown into KNeighborsClassifier's .predict function.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X), sample_weight=sample_weight))
    
    def clone(self):
        return hyb_KNeighborsClassifier(**self.get_params(deep = True))
    


    

class PNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """
    Probabilistic nearest neighbors classification model, from the paper
    "Probabilistic Nearest Neighbors Classification". Probably compatible with
    sklearn.

    Seperate implementation from the R version I made after handing in the thesis.
    Not used inside the thesis. It is more of a proof of concept that it can work
    on full >60000 training points without using 70GB of memory,
    because it does not make a distance matrix like the R version. 
    Feel free to experiment with it.
    
    The R version does automatic k selection by first setting the max k as the first
    r in beta_r that gets 0 from the L-BFGS-B minimization, then doing 5-fold cross validation
    between 1 and that maximum k. This python version does not do any k selection.
    The chosen k is the k that this thing is going to use.


    Parameters
    ----------
    n_neighbors : int, default = 5
        Number of neighbors to use.
    beta_max : float, default = 10.0
        Maximum value of betas fitted during maximizations that happen in fit().
        The minimum is fixed at 0.
    predict_chunksize : int, default = 10000
        During prediction, the test data is split into chunks of approximately this size.
        On each chunk a distance matrix between the points in the chunk and the points in the training
        set is calculated.
    gpu : bool, default = False
        When True the sklearn and numpy present during prediction will be
        replaced by cuml and cupy to enable GPU acceleration.
    
    It also accepts all other arguments that the sklearn's NearestNeighbors accepts.
    """
    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=1.0,
        beta_max=10.0,
        predict_chunksize = 5000,
        gpu = False,
        weights="uniform",
        algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params: dict | None = None,
        n_jobs: int | None = None,
    ):
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.beta_max = beta_max
        self.predict_chunksize = predict_chunksize
        self.gpu = gpu

        if gpu:
            from cuml.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                metric=metric,
                n_jobs=n_jobs,
            )
        else:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                metric_params=metric_params,
                p=p,
                n_jobs=n_jobs,
            )
    
    def set_params(self, **params):
        radius = params.pop("radius", None)
        if radius is not None:
            self.radius = radius
        gpu = params.pop("gpu", None)
        if gpu is not None: self.gpu = gpu
        beta_max = params.pop("beta_max", None)
        if beta_max is not None: self.beta_max = beta_max
        predict_chunksize = params.pop("predict_chunksize", None)
        if predict_chunksize is not None: self.predict_chunksize = predict_chunksize

        if self.gpu:
            # Remove any cuml specific arguments so that mlxtend doesn't complain
            params.pop("leaf_size", None)
            params.pop("metric_params", None)
            params.pop("p", None)

        self.nn.set_params(**params)
        return self
    
    def get_params(self, deep: bool = True) -> dict:
        params = {
            "n_neighbors": self.n_neighbors,
            "radius": self.radius,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": getattr(self, "leaf_size", 30),
            "p": getattr(self, "p", 2),
            "metric": self.metric,
            "metric_params": self.metric_params,
            "n_jobs": self.n_jobs,
        }
        # Remove any cuml specific arguments so that mlxtend doesn't complain
        params.pop("verbose", None)
        params.pop("output_type", None)
        params.pop("algo_params", None)

        params["gpu"] = self.gpu
        params["beta_max"] = self.beta_max
        params["predict_chunksize"] = self.predict_chunksize

        return params

    def fit(self, X, y):
        """Fit the probabilistic nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : PNeighborsClassifier
            The fitted probabilistic nearest neighbors classifier.
        """

        def determine_cycle_counts(out_indices: np.ndarray) -> dict[int, int]:
            """
            Determine number of cycle of all sizes of the neighborhood digraph supplied via indices array,
            necessary for the polynomial time calculation for the normalization constant in the paper.

            Not the algorithm 1 from the paper.
            """
            # Create mask of datapoints where points not forming a cycle is False
            col_ind = out_indices.copy()
            n_samples = len(out_indices)
            unique_mask = np.full(n_samples, True)
            newunique_mask = np.full(n_samples, False)
            # ... by repeatedly masking out points with no indegree
            while True:
                newunique_mask = np.full(n_samples, False)
                newunique_mask[col_ind[unique_mask]] = True
                if np.all(unique_mask == newunique_mask): break
                unique_mask = newunique_mask

            from scipy.sparse import csr_matrix

            # Make adjacency matrix (actually a permutation matrix since each point only has one r-th neighbor)
            row_ind = np.arange(n_samples)
            data = np.full_like(row_ind, 1, int)
            adjacency = csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, n_samples))

            # Keep powering the adjacency matrix and check diagonal to find cycles until
            # all points True in the unique_mask returns to its original point
            # (it shouldn't explode since permutation matrix)
            cur_nn = adjacency
            i = 1
            res: dict[int, int] = {}
            while np.any(unique_mask):
                # divide by i since cycle of length i == i points that comes back to
                # its original point (therefore appear at diagonal)
                count = (cur_nn).diagonal()[unique_mask].astype(bool).sum()//i
                if count:
                    res[i] = count
                # Remove points on found cycles out of consideration by masking them
                # Combined with all non-cycle points masked, the while-loop should terminate when all cycles are found
                unique_mask &= ~(cur_nn).diagonal().astype(bool)
                cur_nn @= adjacency
                i += 1
            
            return res

        def fit_beta_r(feature_count, Y, out_indices, beta_max):
            """
            Find beta_r that maximizes p_r from the paper.
            """
            cycle_counts = determine_cycle_counts(out_indices)
            n = len(Y)
            
            def minlogp_r(beta_r) -> float:
                from scipy.special import logsumexp
                logz_r = (
                    np.log(np.exp(beta_r) + feature_count - 1)*(n - np.sum([m*cmr for (m, cmr) in cycle_counts.items()]))
                    + np.sum([
                            np.log((np.exp(beta_r) + feature_count - 1))*m* cmr for (m, cmr) in cycle_counts.items()
                        ])
                ) if beta_r == 0 else (
                    np.log(np.exp(beta_r) + feature_count - 1)*(n - np.sum([m*cmr for (m, cmr) in cycle_counts.items()]))
                    + np.sum(
                        [
                            logsumexp([
                                np.log((np.exp(beta_r) + feature_count - 1))*m,
                                np.log(feature_count - 1) + np.log(np.exp(beta_r) - 1)*m
                            ])
                            * cmr for (m, cmr) in cycle_counts.items()
                        ])
                )
                return -(beta_r*np.sum(Y == Y[out_indices]) - logz_r)
            
            from scipy.optimize import minimize
            optim_res = minimize(minlogp_r, np.array([1]), bounds=[(0, beta_max)], method="L-BFGS-B")
            return optim_res.x[0]
        

        from sklearn.utils.validation import validate_data
        from sklearn.utils.multiclass import unique_labels
        from joblib import Parallel, delayed
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        if self.gpu:
            import cupy as cp
            self.classes_ = unique_labels(y)
            self.classes_gpu_ = cp.asarray(self.classes_)
            self.X_ = cp.asarray(X)
            self.y_ = cp.asarray(y)
            # Store the classes seen during fit
            self.nn.fit(self.X_)
            # Distances we use later during prediction, rth_neighrbors we feed into fit_beta_r
            (distances, rth_neighbors) = self.nn.kneighbors(return_distance=True)
            self.distances_ = cp.asarray(distances, dtype=cp.float64)
            rth_neighbors = cp.asnumpy(rth_neighbors)
            feature_count = len(self.classes_)
            # Parallel fit beta_r for each r from 1 to k
            betas = Parallel(n_jobs=self.n_jobs if self.n_jobs is not None else 1)(delayed(fit_beta_r)(feature_count, y, rth_neighbors[:, r], self.beta_max) for r in range(self.n_neighbors))
            self.betas_ = cp.asarray(betas)

            
        else:
            self.classes_ = unique_labels(y)
            self.X_ = X
            self.y_ = y
            self.nn.fit(X)
            # Distances we use later during prediction, rth_neighrbors we feed into fit_beta_r
            (self.distances_, rth_neighbors) = self.nn.kneighbors(return_distance=True)
            feature_count = len(self.classes_)
            # Parallel fit beta_r for each r from 1 to k
            betas = Parallel(n_jobs=self.n_jobs)(delayed(fit_beta_r)(feature_count, y, rth_neighbors[:, r], self.beta_max) for r in range(self.n_neighbors))
            self.betas_ = np.array(betas)

        # Mask used dufing predict_p_r
        if self.gpu:
            self._themask_ = self.y_[:, np.newaxis] == self.classes_gpu_[np.newaxis, :]
        else:
            self._themask_ = self.y_[:, np.newaxis] == self.classes_[np.newaxis, :]

        return self
    
    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy.ndarray
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries, n_classes)
            The class probabilities of the input samples. Classes are ordered as given during fit().
        """
        def predict_p_r(classes, beta_r, r: int, X_distances: np.ndarray, ytrain: np.ndarray, y_distances: np.ndarray, y_outinds: np.ndarray):
            """ 
            Calculate p_r with the new points.
            """
            # $\mathbb{I}(y_{n+1}=y_{[n+1]_r})$ in the paper page 9 of 12
            fst_term = (ytrain[y_outinds[:, np.newaxis]] == classes)
            # The sum that appears after the first term
            snd_term = np.sum(
                # 60000 x 2 x 1
                (self._themask_)[:, :, np.newaxis]
                # 60000 x 1 x 10000
                & (
                    # 60000 x 10000
                    ((X_distances[:, r-1, np.newaxis] <= y_distances) & (y_distances < X_distances[:, r, np.newaxis])) if r > 0 else
                    (y_distances < X_distances[:, r, np.newaxis])
                )[:, np.newaxis, :], axis=0
            )
            # Normalize
            p_rs = np.exp(beta_r*((fst_term + snd_term.T)))
            p_rs /= np.sum(p_rs, axis=1, keepdims=True)
            return p_rs
        

        from sklearn.utils.validation import check_is_fitted, check_array
        check_is_fitted(self)

        X = check_array(X)

        num_chunks = int(np.ceil(len(X) / self.predict_chunksize))
        chunks = np.array_split(X, num_chunks)

        from joblib import Parallel, delayed
        p_rlist = []
        if self.gpu:
            import cupy as cp
            from cuml.metrics import pairwise_distances
            from cupyx.scipy.special import logsumexp
            for chunk in chunks:
                # Instead of parallel calculate for each r, we do one big numpy(cupy) array manipulation
                chunk_gpu = cp.asarray(chunk)
                outinds = self.nn.kneighbors(chunk_gpu, return_distance=False)
                # Distances to calculate what points in the original train set sees new points as their r-th neighbor
                # (reverse nearest neighbor) needed for snd_term
                # 60000 x 10000
                y_distances = cp.asarray(pairwise_distances(self.X_, chunk_gpu))

                # $\mathbb{I}(y_{n+1}=y_{[n+1]_r})$ in the paper page 9 of 12
                # 10000 x 50 x 2
                fst_term = (self.y_[outinds[:, :, cp.newaxis]] == self.classes_gpu_)
                # The sum that appears after the first term
                # 2 x 50 x 10000
                snd_term = cp.sum(
                    # 60000 x 2 x 1 x 1
                    (self._themask_)[:, :, cp.newaxis, cp.newaxis]
                    # 60000 x 1 x 50 x 10000
                    & (
                        (
                            # 60000 x 50 x 10000
                            cp.concat(
                                (
                                    cp.full((y_distances.shape[0], 1, y_distances.shape[1]), True), 
                                    (self.distances_[:, :-1, cp.newaxis] <= y_distances[:, cp.newaxis, :])
                                ), axis=1)
                                # 60000 x 50 x 10000
                            & (y_distances[:, cp.newaxis, :] < self.distances_[:, :, cp.newaxis]))
                        )[:, cp.newaxis, :, :], axis=0
                )
                # 10000 x 50 x 2
                scores = (
                    self.betas_[cp.newaxis, :, cp.newaxis]
                    * (fst_term + cp.transpose(snd_term, (2, 1, 0)))
                )

                p_rs = cp.exp(
                    scores - logsumexp(scores, axis=2, keepdims=True)
                )
                # 10000 x 2
                p_rs = cp.mean(p_rs, axis=1)
                p_rlist.append(p_rs)
            
            p_r = cp.vstack(p_rlist)
            return cp.asnumpy(p_r)


        else:
            from sklearn.metrics import pairwise_distances
            from scipy.special import logsumexp
            for chunk in chunks:
                # Distances to calculate what points in the original train set sees new points as their r-th neighbor
                # (reverse nearest neighbor) needed for snd_term
                y_distances = pairwise_distances(self.X_, chunk, n_jobs=self.n_jobs)
                outinds = self.nn.kneighbors(chunk, return_distance=False)

                # parallel compute p_r for each r
                p_rses = Parallel(n_jobs=self.n_jobs)(delayed(predict_p_r)(self.classes_, self.betas_[r], r, self.distances_, self.y_, y_distances, outinds[:, r]) for r in range(self.n_neighbors))
                p_rs = np.mean(np.array(p_rses), axis=0)
                p_rlist.append(p_rs)
            p_r = np.vstack(p_rlist)
            return p_r
    
    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        p_r = self.predict_proba(X)
        return np.argmax(p_r, axis=1)
    
    def clone(self):
        return PNeighborsClassifier(**self.get_params(deep = True))
