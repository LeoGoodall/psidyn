library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(lme4)
library(tibble)
library(patchwork)

`%||%` <- function(x, y) {
  if (!is.null(x) && !is.na(x)) x else y
}

safe_quantile <- function(x, prob) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  unname(quantile(x, prob, na.rm = TRUE, type = 7))
}

impute_and_standardize <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  x[is.na(x)] <- median(x, na.rm = TRUE)
  x_mean <- mean(x)
  x_sd <- sd(x)
  if (!is.finite(x_sd) || x_sd == 0) {
    return(rep(0, length(x)))
  }
  (x - x_mean) / x_sd
}

save_plot <- function(plot_obj, filename, width, height, dpi = 300) {
  plots_dir <- file.path(analysis_dir, "plots")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  pdf_path <- file.path(plots_dir, paste0(filename, ".pdf"))
  tryCatch(
    {
      ggsave(
        filename = pdf_path,
        plot = plot_obj,
        width = width,
        height = height
      )
      return(pdf_path)
    },
    error = function(e) {
      png_path <- file.path(plots_dir, paste0(filename, ".png"))
      ggsave(
        filename = png_path,
        plot = plot_obj,
        width = width,
        height = height,
        dpi = dpi
      )
      return(png_path)
    }
  )
}

selected_lag <- 16
analysis_dir <- "PersuasionForGood/results/analysis"
plots_dir <- file.path(analysis_dir, "plots")

conversation_path <- file.path(analysis_dir, "persuasion_te_analysis_results.csv")
strategy_path <- file.path(analysis_dir, "persuasion_strategy_te_analysis.csv")
timeseries_path <- file.path(analysis_dir, "persuasion_te_timeseries.csv")
dialog_path <- file.path("PersuasionForGood/data", "dialog.csv")
surprisal_dir <- "PersuasionForGood/results/surprisal"
surprisal_timeseries_path <- file.path(surprisal_dir, "surprisal_timeseries.csv")
surprisal_summary_path <- file.path(surprisal_dir, "surprisal_conversation_summary.csv")
convo_info_path <- file.path("PersuasionForGood/data", "convo_info.csv")
full_info_path <- file.path("PersuasionForGood/data", "full_info.csv")

dir.create(analysis_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

strategy_focus <- tibble(
  strategy = c(
    "logical-appeal",
    "emotion-appeal",
    "credibility-appeal",
    "foot-in-the-door",
    "self-modeling",
    "personal-story",
    "donation-information",
    "source-related-inquiry",
    "task-related-inquiry",
    "personal-related-inquiry"
  ),
  strategy_class = c(
    rep("APPEAL", 7),
    rep("INQUIRY", 3)
  )
)

strategy_levels <- strategy_focus$strategy
tercile_levels <- c("early", "mid", "late")

required_files <- c(
  conversation_path,
  strategy_path,
  timeseries_path,
  surprisal_timeseries_path,
  surprisal_summary_path,
  convo_info_path,
  full_info_path
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(
    "Required persuasion analysis files are missing: ",
    paste(missing_files, collapse = ", ")
  )
}

conversation_df <- read_csv(conversation_path, show_col_types = FALSE) %>%
  mutate(
    te_asymmetry = te_persuader_to_persuadee - te_persuadee_to_persuader,
    te_standardized = as.numeric(scale(te_persuader_to_persuadee)),
    success_standardized = as.numeric(scale(persuasion_success))
  )

surprisal_summary_df <- read_csv(surprisal_summary_path, show_col_types = FALSE) %>%
  mutate(
    conversation_id = as.character(conversation_id),
    target_user = as.character(target_user)
  )

surprisal_timeseries_df <- read_csv(surprisal_timeseries_path, show_col_types = FALSE) %>%
  mutate(
    conversation_id = as.character(conversation_id),
    target_post_id = as.character(target_post_id),
    target_user = as.character(target_user)
  )

# Summary statistics
summary_stats <- conversation_df %>%
  summarise(
    total_conversations = n(),
    mean_persuasion_success = mean(persuasion_success, na.rm = TRUE),
    std_persuasion_success = sd(persuasion_success, na.rm = TRUE),
    mean_ste_p2p = mean(te_persuader_to_persuadee, na.rm = TRUE),
    std_ste_p2p = sd(te_persuader_to_persuadee, na.rm = TRUE),
    mean_ste_p2r = mean(te_persuadee_to_persuader, na.rm = TRUE),
    std_ste_p2r = sd(te_persuadee_to_persuader, na.rm = TRUE)
  )
write_csv(summary_stats, file.path(analysis_dir, "persuasion_summary_stats.csv"))

surprisal_role_stats <- surprisal_summary_df %>%
  mutate(
    role = if_else(target_user == "0", "persuader", "persuadee")
  ) %>%
  group_by(role) %>%
  summarise(
    mean_median_surprisal = mean(median_surprisal_bits_per_token, na.rm = TRUE),
    sd_median_surprisal = sd(median_surprisal_bits_per_token, na.rm = TRUE),
    median_median_surprisal = median(median_surprisal_bits_per_token, na.rm = TRUE),
    n_conversations = n(),
    .groups = "drop"
  )

write_csv(
  surprisal_role_stats,
  file.path(analysis_dir, "surprisal_role_summary.csv")
)

# Donation logistic regression
strategy_df <- read_csv(strategy_path, show_col_types = FALSE)
convo_info_df <- read_csv(convo_info_path, show_col_types = FALSE)
full_info_df <- read_csv(full_info_path, show_col_types = FALSE)

persuadee_meta <- convo_info_df %>%
  filter(B4 == 1) %>%
  transmute(
    conversation_id = B2,
    persuadee_id = B3,
    persuadee_actual_donation = as.numeric(B5),
    persuadee_intended_donation = as.numeric(B6)
  )

profile_vars <- c("age.x", "agreeable.x", "care.x", "benevolence.x", "rational.x")

persuadee_profiles <- full_info_df %>%
  filter(B4 == 1) %>%
  transmute(
    conversation_id = B2,
    persuadee_id = B3,
    age.x = as.numeric(age.x),
    agreeable.x = as.numeric(agreeable.x),
    care.x = as.numeric(care.x),
    benevolence.x = as.numeric(benevolence.x),
    rational.x = as.numeric(rational.x)
  )

donation_strategy_counts <- strategy_df %>%
  filter(strategy == "donation-information") %>%
  transmute(
    conversation_id,
    donation_information_count = as.numeric(strategy_count)
  )

donation_model_base <- conversation_df %>%
  select(
    conversation_id,
    te_persuader_to_persuadee,
    te_standardized,
    actual_donation
  ) %>%
  left_join(persuadee_meta, by = "conversation_id") %>%
  left_join(persuadee_profiles, by = c("conversation_id", "persuadee_id")) %>%
  left_join(donation_strategy_counts, by = "conversation_id") %>%
  mutate(
    donation_information_count = coalesce(donation_information_count, 0),
    donation_amount = coalesce(persuadee_actual_donation, actual_donation),
    donated_flag = if_else(donation_amount > 0, 1, 0, missing = NA_real_)
  )

donation_model_data <- donation_model_base %>%
  mutate(
    across(
      all_of(profile_vars),
      impute_and_standardize,
      .names = "{.col}_z"
    ),
    persuadee_id = factor(persuadee_id)
  ) %>%
  filter(!is.na(donated_flag))

if (nrow(donation_model_data) > 0) {
  profile_z_vars <- paste0(profile_vars, "_z")
  donation_formula <- as.formula(
    paste(
      "donated_flag ~ te_standardized + donation_information_count +",
      paste(profile_z_vars, collapse = " + ")
    )
  )
  glmer_formula <- update(donation_formula, . ~ . + (1 | persuadee_id))

  glm_coef_df <- NULL
  glmer_coef_df <- NULL
  fit_stats <- tibble()

  if (n_distinct(donation_model_data$donated_flag) > 1) {
    donation_glm <- glm(
      donation_formula,
      data = donation_model_data,
      family = binomial(link = "logit")
    )
    glm_coef_df <- coef(summary(donation_glm)) %>%
      as.data.frame() %>%
      rownames_to_column("term") %>%
      rename(
        estimate = Estimate,
        std_error = `Std. Error`,
        statistic = `z value`,
        p_value = `Pr(>|z|)`
      ) %>%
      mutate(
        model = "glm",
        n = nobs(donation_glm),
        log_likelihood = as.numeric(logLik(donation_glm)),
        aic = AIC(donation_glm),
        bic = BIC(donation_glm)
      )

    glm_frame <- model.frame(donation_glm)
    glm_outcome <- glm_frame$donated_flag
    fit_stats <- bind_rows(
      fit_stats,
      tibble(
        model = "glm",
        n = length(glm_outcome),
        n_donations = sum(glm_outcome == 1),
        donation_rate = mean(glm_outcome == 1),
        log_likelihood = as.numeric(logLik(donation_glm)),
        aic = AIC(donation_glm),
        bic = BIC(donation_glm)
      )
    )
  } else {
    warning("Insufficient variation in donation outcome for logistic regression.")
  }

  repeated_participants <- donation_model_data %>%
    count(persuadee_id, name = "n_obs") %>%
    filter(n_obs > 1)

  if (nrow(repeated_participants) > 0) {
    donation_glmer <- tryCatch(
      glmer(
        glmer_formula,
        data = donation_model_data,
        family = binomial(link = "logit"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
      ),
      error = function(e) {
        warning("Mixed-effects logistic regression failed: ", conditionMessage(e))
        NULL
      }
    )

    if (!is.null(donation_glmer)) {
      glmer_coef_df <- coef(summary(donation_glmer)) %>%
        as.data.frame() %>%
        rownames_to_column("term") %>%
        rename(
          estimate = Estimate,
          std_error = `Std. Error`,
          statistic = `z value`,
          p_value = `Pr(>|z|)`
        ) %>%
        mutate(
          model = "glmer",
          n = nobs(donation_glmer),
          log_likelihood = as.numeric(logLik(donation_glmer)),
          aic = AIC(donation_glmer),
          bic = BIC(donation_glmer)
        )

      glmer_frame <- model.frame(donation_glmer)
      glmer_outcome <- glmer_frame$donated_flag
      fit_stats <- bind_rows(
        fit_stats,
        tibble(
          model = "glmer",
          n = length(glmer_outcome),
          n_donations = sum(glmer_outcome == 1),
          donation_rate = mean(glmer_outcome == 1),
          log_likelihood = as.numeric(logLik(donation_glmer)),
          aic = AIC(donation_glmer),
          bic = BIC(donation_glmer)
        )
      )
    }
  }

  coefficient_list <- Filter(
    function(df) !is.null(df) && nrow(df) > 0,
    list(glm_coef_df, glmer_coef_df)
  )

  if (length(coefficient_list) > 0) {
    coefficient_results <- bind_rows(coefficient_list)
    write_csv(
      coefficient_results,
      file.path(analysis_dir, "donation_logistic_coefficients.csv")
    )
  }

  if (nrow(fit_stats) > 0) {
    write_csv(
      fit_stats,
      file.path(analysis_dir, "donation_logistic_model_fit.csv")
    )
  }

  donation_summary <- donation_model_data %>%
    summarise(
      n_conversations = n(),
      n_unique_persuadees = n_distinct(persuadee_id),
      n_donations = sum(donated_flag == 1),
      donation_rate = mean(donated_flag == 1),
      median_donation = median(donation_amount, na.rm = TRUE),
      mean_donation = mean(donation_amount, na.rm = TRUE)
    )

  write_csv(
    donation_summary,
    file.path(analysis_dir, "donation_logistic_outcome_summary.csv")
  )
} else {
  warning("No usable rows for donation logistic analysis; skipping.")
}

# Strategy effects analysis (Hedges' g)
strategy_presence <- strategy_df %>%
  filter(strategy %in% strategy_levels) %>%
  mutate(strategy_used = strategy_count > 0) %>%
  select(conversation_id, strategy, strategy_used)

strategy_analysis_base <- expand_grid(
  conversation_id = conversation_df$conversation_id,
  strategy = strategy_levels
) %>%
  left_join(strategy_presence, by = c("conversation_id", "strategy")) %>%
  mutate(strategy_used = coalesce(strategy_used, FALSE)) %>%
  left_join(strategy_focus, by = "strategy") %>%
  left_join(
    conversation_df %>%
      select(
        conversation_id,
        te_persuader_to_persuadee,
        te_persuadee_to_persuader,
        te_asymmetry,
        persuasion_success
      ),
    by = "conversation_id"
  )

hedges_g_metrics <- function(x_used, x_not) {
  x_used <- x_used[!is.na(x_used)]
  x_not <- x_not[!is.na(x_not)]
  n_used <- length(x_used)
  n_not <- length(x_not)
  mean_used <- if (n_used > 0) mean(x_used) else NA_real_
  mean_not <- if (n_not > 0) mean(x_not) else NA_real_
  sd_used <- if (n_used > 1) sd(x_used) else NA_real_
  sd_not <- if (n_not > 1) sd(x_not) else NA_real_
  pooled_var <- ((n_used - 1) * (sd_used %||% 0)^2 + (n_not - 1) * (sd_not %||% 0)^2) /
    (n_used + n_not - 2)
  pooled_sd <- if (is.finite(pooled_var) && pooled_var > 0) sqrt(pooled_var) else NA_real_
  hedges_g <- NA_real_
  hedges_g_se <- NA_real_
  ci_lower <- NA_real_
  ci_upper <- NA_real_
  if (!is.na(pooled_sd) && pooled_sd > 0 && n_used > 1 && n_not > 1) {
    d <- (mean_used - mean_not) / pooled_sd
    correction <- 1 - (3 / (4 * (n_used + n_not) - 9))
    hedges_g <- d * correction
    hedges_g_se <- sqrt(
      ((n_used + n_not) / (n_used * n_not)) +
        ((hedges_g^2) / (2 * (n_used + n_not - 2)))
    )
    ci_lower <- hedges_g - 1.96 * hedges_g_se
    ci_upper <- hedges_g + 1.96 * hedges_g_se
  }
  list(
    n_conversations_used = n_used,
    n_conversations_not_used = n_not,
    mean_ste_when_used = mean_used,
    std_ste_when_used = sd_used,
    mean_ste_when_not_used = mean_not,
    std_ste_when_not_used = sd_not,
    ste_difference = mean_used - mean_not,
    hedges_g = hedges_g,
    hedges_g_se = hedges_g_se,
    hedges_g_ci_lower = ci_lower,
    hedges_g_ci_upper = ci_upper,
    log_ratio_mean = if (!is.na(mean_used) && !is.na(mean_not) && mean_used > 0 && mean_not > 0) {
      log(mean_used / mean_not)
    } else {
      NA_real_
    }
  )
}

compute_strategy_effects <- function(x_used, x_not) {
  base_stats <- hedges_g_metrics(x_used, x_not)
  t_res <- if (base_stats$n_conversations_used > 1 && base_stats$n_conversations_not_used > 1) {
    tryCatch(
      t.test(x_used, x_not, var.equal = FALSE),
      error = function(e) NULL
    )
  } else {
    NULL
  }
  w_res <- if (base_stats$n_conversations_used > 0 && base_stats$n_conversations_not_used > 0) {
    tryCatch(
      wilcox.test(x_used, x_not, exact = FALSE),
      error = function(e) NULL
    )
  } else {
    NULL
  }
  c(
    base_stats,
    list(
      t_statistic = if (!is.null(t_res)) unname(t_res$statistic) else NA_real_,
      t_p_value = if (!is.null(t_res)) t_res$p.value else NA_real_,
      mannwhitney_u = if (!is.null(w_res)) unname(w_res$statistic) else NA_real_,
      mannwhitney_p_value = if (!is.null(w_res)) w_res$p.value else NA_real_
    )
  )
}

strategy_effects <- strategy_analysis_base %>%
  group_by(strategy, strategy_class) %>%
  group_modify(~ {
    ste_used <- .x$te_persuader_to_persuadee[.x$strategy_used]
    ste_not <- .x$te_persuader_to_persuadee[!.x$strategy_used]
    as_tibble(compute_strategy_effects(ste_used, ste_not))
  }) %>%
  ungroup() %>%
  arrange(desc(hedges_g))

write_csv(strategy_effects, file.path(analysis_dir, "persuasion_strategy_analysis_stats.csv"))

# Strategy indicators for LME model
strategy_indicator_cols <- paste0(
  "strategy_",
  str_replace_all(strategy_levels, "-", "_")
)

strategy_flags <- strategy_analysis_base %>%
  select(conversation_id, strategy, strategy_used) %>%
  distinct() %>%
  pivot_wider(
    names_from = strategy,
    values_from = strategy_used,
    values_fill = FALSE
  )

col_mapping <- strategy_indicator_cols
names(col_mapping) <- strategy_levels

strategy_wide <- strategy_flags %>%
  mutate(across(-conversation_id, as.integer)) %>%
  rename_with(~ col_mapping[.x], -conversation_id)

appeal_cols <- col_mapping[strategy_focus$strategy_class == "APPEAL"]
inquiry_cols <- col_mapping[strategy_focus$strategy_class == "INQUIRY"]

missing_strategy_cols <- setdiff(strategy_indicator_cols, names(strategy_wide))
if (length(missing_strategy_cols) > 0) {
  strategy_wide[missing_strategy_cols] <- 0L
}

strategy_wide <- strategy_wide %>%
  mutate(across(all_of(strategy_indicator_cols), ~coalesce(., 0L))) %>%
  mutate(
    appeal_any = as.integer(rowSums(select(., all_of(appeal_cols))) > 0),
    inquiry_any = as.integer(rowSums(select(., all_of(inquiry_cols))) > 0)
  )

# Timeseries data preparation
timeseries_df <- read_csv(timeseries_path, show_col_types = FALSE) %>%
  mutate(
    target_post_id = as.character(target_post_id),
    conversation_id = as.character(conversation_id)
  ) %>%
  left_join(
    surprisal_timeseries_df %>%
      select(
        conversation_id,
        target_post_id,
        surprisal_bits_per_token
      ),
    by = c("conversation_id", "target_post_id")
  )

if ("lag" %in% names(timeseries_df)) {
  timeseries_df <- timeseries_df %>% filter(is.na(lag) | lag == selected_lag)
}

meta_df <- read_csv(
  dialog_path,
  show_col_types = FALSE,
  col_types = cols(
    post_id = col_character(),
    B2 = col_character(),
    B4 = col_double(),
    Turn = col_double(),
    Unit = col_character()
  )
) %>%
  transmute(
    target_post_id = post_id,
    conversation_id = B2,
    target_role = if_else(B4 == 0, "persuader", "persuadee"),
    turn_length_words = if_else(is.na(Unit), 0L, lengths(str_split(Unit, "\\s+")))
  )

analysis_df <- timeseries_df %>%
  left_join(meta_df, by = c("target_post_id", "conversation_id")) %>%
  mutate(
    source_role = if_else(as.character(source_user) == "0", "persuader", "persuadee"),
    Condition = factor(source_role, levels = c("persuader", "persuadee")),
    inv_tokens = if_else(turn_length_words > 0, 1 / turn_length_words, NA_real_),
    inv_tokens_sq = inv_tokens^2,
    surprisal_bits_per_token = as.numeric(surprisal_bits_per_token)
  ) %>%
  left_join(strategy_wide, by = "conversation_id") %>%
  mutate(
    across(all_of(c(strategy_indicator_cols, "appeal_any", "inquiry_any")), ~coalesce(., 0)),
    te_bits_per_token = as.numeric(te_bits_per_token)
  ) %>%
  filter(
    !is.na(te_bits_per_token),
    !is.na(surprisal_bits_per_token),
    !is.na(Condition),
    !is.na(inv_tokens),
    is.finite(inv_tokens),
    is.finite(te_bits_per_token),
    is.finite(surprisal_bits_per_token)
  )

# Linear mixed-effects model
lme_output_path <- file.path(analysis_dir, "persuasion_te_length_lme_summary.txt")

strategy_terms <- c("appeal_any", "inquiry_any", strategy_indicator_cols)
fixed_terms <- c("Condition", "inv_tokens", "inv_tokens_sq", "surprisal_bits_per_token", strategy_terms)
lme_formula <- as.formula(
  paste(
    "te_bits_per_token ~",
    paste(fixed_terms, collapse = " + "),
    "+ (1 | conversation_id)"
  )
)

if (nrow(analysis_df) > 0) {
  lme_data <- analysis_df %>%
    select(
      conversation_id,
      te_bits_per_token,
      Condition,
      inv_tokens,
      inv_tokens_sq,
      surprisal_bits_per_token,
      all_of(strategy_terms)
    )

  model <- lmer(
    lme_formula,
    data = lme_data,
    REML = FALSE
  )

  sink(lme_output_path)
  cat(
    "Linear mixed-effects model (lag = ",
    selected_lag,
    "):\n",
    deparse(lme_formula),
    "\n\n",
    sep = ""
  )
  print(summary(model))
  sink()

  fixed_effects_df <- coef(summary(model)) %>%
    as.data.frame() %>%
    rownames_to_column("term")
  write_csv(
    fixed_effects_df,
    file.path(analysis_dir, "persuasion_te_length_lme_fixed_effects.csv")
  )
} else {
  warning("No valid rows available for mixed-effects modelling; skipping model fit.")
}

# Timeseries plot preparation
timeseries_ordered <- if ("target_timestamp" %in% names(timeseries_df)) {
  timeseries_df %>%
    arrange(conversation_id, target_timestamp, target_post_id)
} else {
  timeseries_df %>%
    arrange(conversation_id, target_post_id)
}

timeseries_enriched <- timeseries_ordered %>%
  group_by(conversation_id) %>%
  mutate(
    turn_index = row_number(),
    total_turns = n(),
    turn_fraction = if_else(
      total_turns > 1,
      (turn_index - 1) / (total_turns - 1),
      0
    ),
    tercile = case_when(
      turn_fraction <= 1 / 3 ~ "early",
      turn_fraction <= 2 / 3 ~ "mid",
      TRUE ~ "late"
    )
  ) %>%
  ungroup() %>%
  mutate(tercile = factor(tercile, levels = tercile_levels))

fraction_bin_count <- 50
fraction_breaks <- seq(0, 1, length.out = fraction_bin_count + 1)
fraction_midpoints <- head(fraction_breaks, -1) + diff(fraction_breaks) / 2
direction_colors <- c("persuader->persuadee" = "#d95f02", "persuadee->persuader" = "#1b9e77")
direction_levels <- names(direction_colors)
direction_labels <- c(
  "persuader->persuadee" = "Persuader -> Persuadee",
  "persuadee->persuader" = "Persuadee -> Persuader"
)

directional_timeseries <- timeseries_enriched %>%
  mutate(
    direction = case_when(
      as.character(source_user) == "0" & as.character(target_user) == "1" ~ "persuader->persuadee",
      as.character(source_user) == "1" & as.character(target_user) == "0" ~ "persuadee->persuader",
      TRUE ~ NA_character_
    ),
    fraction_bin = cut(
      turn_fraction,
      breaks = fraction_breaks,
      include_lowest = TRUE,
      labels = fraction_midpoints
    )
  ) %>%
  filter(!is.na(direction), !is.na(fraction_bin)) %>%
  mutate(fraction_bin = as.numeric(as.character(fraction_bin))) %>%
  group_by(direction, fraction_bin) %>%
  summarise(
    median_ste = median(te_bits_per_token, na.rm = TRUE),
    q25 = safe_quantile(te_bits_per_token, 0.25),
    q75 = safe_quantile(te_bits_per_token, 0.75),
    median_surprisal = median(surprisal_bits_per_token, na.rm = TRUE),
    n_turns = n(),
    .groups = "drop"
  ) %>%
  mutate(direction = factor(direction, levels = direction_levels))

write_csv(
  directional_timeseries,
  file.path(analysis_dir, "directional_te_timeseries_summary.csv")
)

# Directional timeseries plot (with surprisal overlay)
directional_ts_plot <- NULL
timeseries_line_long <- directional_timeseries %>%
  select(direction, fraction_bin, median_ste, median_surprisal) %>%
  pivot_longer(
    cols = c(median_ste, median_surprisal),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(metric, median_ste = "STE", median_surprisal = "Surprisal"),
    metric = factor(metric, levels = c("STE", "Surprisal"))
  )

if (nrow(directional_timeseries) > 0) {
  directional_ts_plot <- ggplot() +
    geom_ribbon(
      data = directional_timeseries,
      aes(x = fraction_bin, ymin = q25, ymax = q75, fill = direction),
      alpha = 0.12,
      colour = NA
    ) +
    geom_line(
      data = timeseries_line_long,
      aes(
        x = fraction_bin,
        y = value,
        colour = direction,
        linetype = metric
      ),
      linewidth = 0.9
    ) +
    scale_colour_manual(
      values = direction_colors,
      breaks = direction_levels,
      labels = direction_labels
    ) +
    scale_fill_manual(
      values = direction_colors,
      breaks = direction_levels,
      labels = direction_labels,
      guide = "none"
    ) +
    scale_linetype_manual(
      values = c("STE" = "solid", "Surprisal" = "dotted"),
      guide = "none"
    ) +
    labs(
      x = "Conversation progress",
      y = "Bits per token",
      title = "Directional STE and surprisal over conversation turns",
      subtitle = "STE (solid) vs. surprisal (dotted); shaded band = STE IQR"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  save_plot(directional_ts_plot, "directional_ste_timeseries", width = 7, height = 4.5)
}

# Strategy Hedges' g plot
strategy_effect_plot <- NULL
if (nrow(strategy_effects) > 0) {
  strategy_effect_plot <- strategy_effects %>%
    mutate(
      strategy_label = str_replace_all(strategy, "-", " "),
      strategy_label = factor(
        strategy_label,
        levels = strategy_label[order(hedges_g, decreasing = TRUE)]
      )
    ) %>%
    ggplot(aes(x = hedges_g, y = strategy_label, fill = strategy_class)) +
    geom_col(alpha = 0.8) +
    geom_errorbarh(
      aes(xmin = hedges_g_ci_lower, xmax = hedges_g_ci_upper),
      height = 0.25,
      colour = "#444444"
    ) +
    scale_fill_manual(
      values = c("APPEAL" = "#4A4A4A", "INQUIRY" = "#A0A0A0"),
      breaks = c("APPEAL", "INQUIRY")
    ) +
    labs(
      x = "Hedges' g (effect size)",
      y = NULL,
      fill = "Strategy class",
      title = "Strategy influence on STE (persuader -> persuadee)",
      subtitle = "Negative values indicate tactics that dampen persuader influence"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  save_plot(strategy_effect_plot, "strategy_hedges_g", width = 7, height = 4.5)
}

# Combined figure (2 panels only)
if (!is.null(directional_ts_plot) && !is.null(strategy_effect_plot)) {
  combined_plot <- directional_ts_plot | strategy_effect_plot

  # Save to results folder
  results_dir <- "PersuasionForGood/results"
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(
    filename = file.path(results_dir, "persuasion_results.pdf"),
    plot = combined_plot,
    width = 14,
    height = 5
  )
}
