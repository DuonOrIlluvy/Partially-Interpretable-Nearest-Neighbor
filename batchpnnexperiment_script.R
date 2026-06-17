# Script that does hybrid pnn experiment for all 9 datasets x blackboxes,
# with forward and backward sequential feature selection at once

# The pnnclass from https://github.com/paulocmarquesf/pnnclass/tree/master crashes with segmentation fault.
# You need to download the source code, change the line 17 of the R/pnnclass.R into
# "if (opt$par == 0 && r != 1) break" and install that using install.packages.
library(pnnclass)
library(foreach)
library(doParallel)
library(dplyr)
library(pracma)

# Forward sequential feature selection of pnn (no black-box involvement for this process)
hybrid_forward_pnn <- function(
  dftrain, dfvalidate, dftest, Ybtest
) {
  # Function to combine each foreach return value
  keep_best <- function(acc, new_val) {
    if (
      mean(acc$y_hat == dfvalidate[["Y"]])
      < mean(new_val$y_hat == dfvalidate[["Y"]])
    ) {
      acc <- new_val
    }
    acc
  }

  all_features <- setdiff(colnames(dftrain), "Y")
  selected <- character(0)
  remaining <- all_features
  max_features <- length(all_features)
  history <- data.frame()

  for (step in seq_len(max_features)) {
    # parallel compute best feature to add
    best_res <- foreach(
      feature = remaining,
      .combine = "keep_best",
      .packages = c("pnnclass"),
      .errorhandling = "remove"
    ) %dopar% {
      print(sprintf("feature %d/%d, trying %s", step, max_features, feature))
      trial_features <- c(selected, feature)
      # Formula about predicting Y using currently selected featuers
      fml <- as.formula(
        paste("Y", "~", paste(trial_features, collapse = " + "))
      )
      # Use pnnclass: train using first 1000 datapoints and test by predicting next 1000 datapoints
      model <- pnnclass(fml, data = dftrain)
      pred <- predict(model, newdata = dfvalidate)
      ret_list <- list(y_hat = matrix(as.integer(pred$y_hat)),
                       added_feature = feature,
                       model = model)
      ret_list
    }
    # Select best performed feature in terms of prediction score of dfvalidate
    selected <- c(selected, best_res$added_feature)
    remaining <- setdiff(remaining, best_res$added_feature)

    pred <- predict(best_res$model, newdata = dftest)
    print(
      sprintf(
        "added feature: %s, accuracy: %f",
        best_res$added_feature,
        mean(pred$y_hat == dftest[["Y"]])
      )
    )

    test_num <- length(dftest[["Y"]])
    # For every feature subset, experiment with resulting probability based hybrid nearest neighbors
    for (offload_thrs in c(linspace(0, 0.5, 20), 0.6)) {
      # If probability of one class is too close to 0.5 (determined by offload_thrs) then defer to black-box
      hyb_yhat <- pred$y_hat
      mask <- abs(pred$prob[, 1] - 0.5) < offload_thrs
      covered_amount <- test_num - sum(mask)
      hyb_yhat[mask] <- Ybtest[[1]][mask]

      history <- bind_rows(
        history,
        data.frame(
          offload_thrs = offload_thrs,
          transparency = covered_amount / test_num,
          accuracy = mean(hyb_yhat == dftest[["Y"]]),
          features_used = step,
          k_nearest = best_res$model$k_hat
        )
      )
    }
  }

  history
}

# Backward sequential feature selection of pnn (no black-box involvement for this process)
hybrid_backward_pnn <- function(
  dftrain, dfvalidate, dftest, Ybtest
) {
  # Function to combine each foreach return value
  keep_best <- function(acc, new_val) {
    if (
      mean(acc$y_hat == dfvalidate[["Y"]])
      < mean(new_val$y_hat == dfvalidate[["Y"]])
    ) {
      acc <- new_val
    }
    acc
  }

  all_features <- setdiff(colnames(dftrain), "Y")
  selected <- character(0)
  remaining <- all_features
  max_features <- length(all_features)
  history <- data.frame()

  # Once with all features
  print(sprintf("feature %d/1, with all features",max_features))
  trial_features <- all_features
  fml <- as.formula(
    paste("Y", "~", paste(trial_features, collapse = " + "))
  )
  model <- pnnclass(fml, data = dftrain)
  pred <- predict(model, newdata = dfvalidate)
  best_res <- list(y_hat = matrix(as.integer(pred$y_hat)),
                    removed_feature = NA,
                    model = model)
  pred <- predict(best_res$model, newdata = dftest)
  print(
    sprintf(
      "Full feature, accuracy: %f",
      mean(pred$y_hat == dftest[["Y"]])
    )
  )
  test_num <- length(dftest[["Y"]])

  # For every feature subset, experiment with resulting probability based hybrid nearest neighbors
  for (offload_thrs in c(linspace(0, 0.5, 20), 0.6)) {
    # If probability of one class is too close to 0.5 (determined by offload_thrs) then defer to black-box
    hyb_yhat <- pred$y_hat
    mask <- abs(pred$prob[, 1] - 0.5) < offload_thrs
    covered_amount <- test_num - sum(mask)
    hyb_yhat[mask] <- Ybtest[[1]][mask]

    history <- bind_rows(
      history,
      data.frame(
        offload_thrs = offload_thrs,
        transparency = covered_amount / test_num,
        accuracy = mean(hyb_yhat == dftest[["Y"]]),
        features_used = max_features,
        k_nearest = best_res$model$k_hat
      )
    )
  }

  for (step in seq_len(max_features - 1)) {
    # parallel compute best feature to add
    best_res <- foreach(
      feature = remaining,
      .combine = "keep_best",
      .packages = c("pnnclass"),
      .errorhandling = "remove"
    ) %dopar% {
      print(sprintf("feature %d/1, trying removing %s",
                    max_features - step, feature))
      trial_features <- setdiff(all_features, c(selected, feature))
      # Formula about predicting Y using currently selected featuers
      fml <- as.formula(
        paste("Y", "~", paste(trial_features, collapse = " + "))
      )
      model <- pnnclass(fml, data = dftrain)
      pred <- predict(model, newdata = dfvalidate)
      ret_list <- list(y_hat = matrix(as.integer(pred$y_hat)),
                       removed_feature = feature,
                       model = model)
      ret_list
    }
    selected <- c(selected, best_res$removed_feature)
    remaining <- setdiff(remaining, best_res$removed_feature)

    pred <- predict(best_res$model, newdata = dftest)
    print(
      sprintf(
        "removed feature: %s, accuracy: %f",
        best_res$removed_feature,
        mean(pred$y_hat == dftest[["Y"]])
      )
    )

    test_num <- length(dftest[["Y"]])
    # For every feature subset, experiment with resulting probability based hybrid nearest neighbors
    for (offload_thrs in c(linspace(0, 0.5, 20), 0.6)) {
      # If probability of one class is too close to 0.5 (determined by offload_thrs) then defer to black-box
      hyb_yhat <- pred$y_hat
      mask <- abs(pred$prob[, 1] - 0.5) < offload_thrs
      covered_amount <- test_num - sum(mask)
      hyb_yhat[mask] <- Ybtest[[1]][mask]

      history <- bind_rows(
        history,
        data.frame(
          offload_thrs = offload_thrs,
          transparency = covered_amount / test_num,
          accuracy = mean(hyb_yhat == dftest[["Y"]]),
          features_used = max_features - step,
          k_nearest = best_res$model$k_hat
        )
      )
    }
  }

  history
}


# to activate %dopar%
cl <- makeCluster(parallel::detectCores() - 1, outfile = "")
registerDoParallel(cl)

blackboxes <- c("rf", "ab", "xg")
path_name <- list(
  c("./datasets/census/", "census"),
  c("./datasets/coupon/", "coupon"),
  c("./datasets/stop&frisk/", "stop-and-frisk")
)

for (blackbox in blackboxes) {
  for (pn in path_name) {
    path <- pn[1]
    name <- pn[2]
    dftrain <- read.csv(paste0(path, "train.csv"))
    dftest <- read.csv(paste0(path, "test.csv"))
    Ybtest <- read.csv(paste0(path, blackbox, "_ybtest.csv"))
    # Convert column names to something R can manage
    # (since education-num == "education minus num" by R formula)
    colnames(dftrain) <- make.names(colnames(dftrain), unique = TRUE)
    colnames(dftest)  <- make.names(colnames(dftest), unique = TRUE)
    # Categorical (0 and 1) columns are considered integer,
    # convert them to numeric so that pnnclass doesn't complain
    x_cols <- setdiff(colnames(dftrain), "Y")
    dftrain[x_cols] <- lapply(dftrain[x_cols], as.numeric)
    dftest[x_cols]  <- lapply(dftest[x_cols],  as.numeric)

    # Proceed with forward selection experiment if result file does not already exist
    forward_name <- paste0(path, name, "_", blackbox, "_hybpnnsc2000fwgreedy_results.csv")
    print(forward_name)
    if (!file.exists(forward_name)) {
      forward_result <- hybrid_forward_pnn(
        dftrain[1:1000, ], dftrain[1001:2000, ], dftest, Ybtest
      )
      write.csv(
        forward_result,
        file = forward_name,
        quote = FALSE,
        row.names = FALSE
      )
    }

    # Proceed with backward selection experiment if result file does not already exist
    backward_name <- paste0(path, name, "_", blackbox, "_hybpnnsc2000bcgreedy_results.csv")
    print(backward_name)
    if (TRUE) {
      backward_result <- hybrid_backward_pnn(
        dftrain[1:1000, ], dftrain[1001:2000, ], dftest, Ybtest
      )
      write.csv(
        backward_result,
        file = backward_name,
        quote = FALSE,
        row.names = FALSE
      )
    }
  }
}

stopCluster(cl)