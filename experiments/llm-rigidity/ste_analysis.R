# Semantic Transfer Entropy Analysis: Cognitive Rigidity vs Flexibility
# Tests hypothesis that flexible targets have higher incoming STE than rigid targets
# Excludes first 5 turns due to lag window = 5

library(lme4)
library(lmerTest)
library(emmeans)

# Load timeseries data (contains per-turn STE values)
data <- read.csv("results/te_timeseries_full.csv", stringsAsFactors = FALSE)

# Filter out turns <= 5 (lag window = 5, so early turns are unreliable)
data <- data[data$turn > 5, ]

# Normalise speaker labels (A/B vs Speaker A/Speaker B)
data$source_user <- gsub("Speaker ", "", data$source_user)
data$target_user <- gsub("Speaker ", "", data$target_user)
data$direction <- paste0(data$source_user, "_to_", data$target_user)

# Create source_type and target_type based on condition and direction
# rigid-rigid: A=rigid, B=rigid
# flexible-flexible: A=flexible, B=flexible
# rigid-flexible: A=rigid, B=flexible
data$source_type <- NA
data$target_type <- NA

# rigid-rigid: both are rigid
data$source_type[data$condition == "rigid-rigid"] <- "rigid"
data$target_type[data$condition == "rigid-rigid"] <- "rigid"

# flexible-flexible: both are flexible
data$source_type[data$condition == "flexible-flexible"] <- "flexible"
data$target_type[data$condition == "flexible-flexible"] <- "flexible"

# rigid-flexible: A is rigid, B is flexible
rf_idx <- data$condition == "rigid-flexible"
data$source_type[rf_idx & data$source_user == "A"] <- "rigid"
data$target_type[rf_idx & data$source_user == "A"] <- "flexible"
data$source_type[rf_idx & data$source_user == "B"] <- "flexible"
data$target_type[rf_idx & data$source_user == "B"] <- "rigid"

# Remove rows with missing type assignments
data <- data[!is.na(data$source_type) & !is.na(data$target_type), ]

# Create directional condition (source_type -> target_type)
data$direction_type <- paste0(
  substr(data$source_type, 1, 1), 
  "_to_", 
  substr(data$target_type, 1, 1)
)

# Rename te column for consistency
data$te_value <- data$te_bits_per_token

# Factor variables
data$source_type <- factor(data$source_type, levels = c("rigid", "flexible"))
data$target_type <- factor(data$target_type, levels = c("rigid", "flexible"))
data$direction_type <- factor(data$direction_type)
data$topic <- factor(data$topic)
data$condition <- factor(data$condition)

# Aggregate to conversation level for some analyses
data_conv <- aggregate(te_value ~ conversation_id + condition + topic + direction + 
                       source_type + target_type + direction_type,
                       data = data, FUN = mean)

# Open output file
sink("results/ste_statistics.txt")

cat("SEMANTIC TRANSFER ENTROPY ANALYSIS\n")

cat("Hypothesis: Flexible targets receive higher incoming STE than rigid targets\n")
cat("(Flexible agents integrate information from partners; rigid agents do not)\n\n")

cat("NOTE: Excluded turns 1-5 due to lag window = 5\n\n")

# Descriptive statistics
cat("1. DESCRIPTIVE STATISTICS\n\n")

cat("Total turn-level observations:", nrow(data), "\n")
cat("Total conversation-direction pairs:", nrow(data_conv), "\n")
cat("Turn range:", min(data$turn), "-", max(data$turn), "\n\n")

cat("Sample sizes by direction type (source -> target):\n")
print(table(data_conv$direction_type))
cat("\n")

cat("Mean STE by direction type (conversation-level):\n")
agg_dir <- aggregate(te_value ~ direction_type, data = data_conv, 
                     FUN = function(x) round(c(mean = mean(x), sd = sd(x), n = length(x)), 4))
print(agg_dir)
cat("\n")

cat("Mean STE by target type:\n")
agg_target <- aggregate(te_value ~ target_type, data = data_conv,
                        FUN = function(x) round(c(mean = mean(x), sd = sd(x), n = length(x)), 4))
print(agg_target)
cat("\n")

cat("Mean STE by source type:\n")
agg_source <- aggregate(te_value ~ source_type, data = data_conv,
                        FUN = function(x) round(c(mean = mean(x), sd = sd(x), n = length(x)), 4))
print(agg_source)
cat("\n")

cat("Mean STE by condition and direction:\n")
agg_cond <- aggregate(te_value ~ condition + direction, data = data_conv,
                      FUN = function(x) round(c(mean = mean(x), sd = sd(x), n = length(x)), 4))
print(agg_cond)
cat("\n")

# Mixed effects model: full model with 4 direction types
cat("2. MIXED EFFECTS MODEL: 4 DIRECTION TYPES\n\n")
cat("Model: te_value ~ direction_type + (1|conversation_id) + (1|topic)\n\n")

model_dir <- lmer(te_value ~ direction_type + (1|conversation_id) + (1|topic), 
                  data = data_conv, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa"))

cat("Fixed effects:\n")
print(round(summary(model_dir)$coefficients, 4))
cat("\n")

cat("Random effects:\n")
print(VarCorr(model_dir))
cat("\n")

cat("ANOVA (Type III):\n")
print(anova(model_dir, type = 3))
cat("\n")

cat("Pairwise comparisons (Tukey):\n")
emm_dir <- emmeans(model_dir, "direction_type")
cat("\nEstimated marginal means:\n")
print(summary(emm_dir))
cat("\nPairwise contrasts:\n")
print(pairs(emm_dir, adjust = "tukey"))
cat("\n")

# Main hypothesis test: target type effect
cat("3. MAIN HYPOTHESIS TEST: TARGET TYPE EFFECT\n\n")
cat("Model: te_value ~ target_type + (1|conversation_id) + (1|topic)\n\n")

model_target <- lmer(te_value ~ target_type + (1|conversation_id) + (1|topic),
                     data = data_conv, REML = FALSE,
                     control = lmerControl(optimizer = "bobyqa"))

cat("Fixed effects:\n")
print(round(summary(model_target)$coefficients, 4))
cat("\n")

cat("ANOVA (Type III):\n")
print(anova(model_target, type = 3))
cat("\n")

cat("Estimated marginal means:\n")
emm_target <- emmeans(model_target, "target_type")
print(summary(emm_target))
cat("\nContrast (flexible - rigid target):\n")
print(pairs(emm_target))
cat("\n")

# Full factorial model
cat("4. FULL FACTORIAL MODEL: SOURCE x TARGET TYPE\n\n")
cat("Model: te_value ~ source_type * target_type + (1|conversation_id) + (1|topic)\n\n")

model_full <- lmer(te_value ~ source_type * target_type + (1|conversation_id) + (1|topic),
                   data = data_conv, REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa"))

cat("Fixed effects:\n")
print(round(summary(model_full)$coefficients, 4))
cat("\n")

cat("ANOVA (Type III):\n")
print(anova(model_full, type = 3))
cat("\n")

# Focused analysis: rigid-flexible condition only
cat("5. FOCUSED ANALYSIS: RIGID-FLEXIBLE CONDITION ONLY\n\n")
cat("This isolates r->f vs f->r comparison within mixed dyads\n\n")

data_rf <- data_conv[data_conv$condition == "rigid-flexible", ]

cat("Sample sizes:\n")
print(table(data_rf$direction_type))
cat("\n")

cat("Mean STE by direction:\n")
agg_rf <- aggregate(te_value ~ direction_type, data = data_rf,
                    FUN = function(x) round(c(mean = mean(x), sd = sd(x), n = length(x)), 4))
print(agg_rf)
cat("\n")

model_rf <- lmer(te_value ~ direction_type + (1|conversation_id) + (1|topic),
                 data = data_rf, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

cat("Model: te_value ~ direction_type + (1|conversation_id) + (1|topic)\n")
cat("Fixed effects:\n")
print(round(summary(model_rf)$coefficients, 4))
cat("\n")

cat("ANOVA:\n")
print(anova(model_rf, type = 3))
cat("\n")

# Paired t-test within conversations
cat("Paired comparison within conversations (r->f vs f->r):\n")
rf_wide <- reshape(data_rf[, c("conversation_id", "direction_type", "te_value")],
                   idvar = "conversation_id", timevar = "direction_type",
                   direction = "wide")
names(rf_wide) <- c("conversation_id", "f_to_r", "r_to_f")
rf_wide <- rf_wide[complete.cases(rf_wide), ]

paired_t <- t.test(rf_wide$r_to_f, rf_wide$f_to_r, paired = TRUE)
cat(sprintf("  N pairs = %d\n", nrow(rf_wide)))
cat(sprintf("  t = %.4f, df = %d, p = %.6f\n", 
            paired_t$statistic, paired_t$parameter, paired_t$p.value))
cat(sprintf("  Mean r->f: %.4f\n", mean(rf_wide$r_to_f)))
cat(sprintf("  Mean f->r: %.4f\n", mean(rf_wide$f_to_r)))
cat(sprintf("  Difference (r->f - f->r): %.4f\n", paired_t$estimate))
cat("\n")

# Effect sizes
cat("6. EFFECT SIZES\n\n")

rigid_target <- data_conv$te_value[data_conv$target_type == "rigid"]
flex_target <- data_conv$te_value[data_conv$target_type == "flexible"]

pooled_sd <- sqrt(((length(rigid_target) - 1) * var(rigid_target) + 
                   (length(flex_target) - 1) * var(flex_target)) / 
                  (length(rigid_target) + length(flex_target) - 2))

cohens_d_target <- (mean(flex_target) - mean(rigid_target)) / pooled_sd

cat("Cohen's d for target type effect (flexible - rigid):\n")
cat(sprintf("  d = %.4f\n\n", cohens_d_target))

cat("Cohen's d for direction type contrasts:\n")
dir_levels <- levels(data_conv$direction_type)
for (i in 1:(length(dir_levels)-1)) {
  for (j in (i+1):length(dir_levels)) {
    g1 <- data_conv$te_value[data_conv$direction_type == dir_levels[i]]
    g2 <- data_conv$te_value[data_conv$direction_type == dir_levels[j]]
    ps <- sqrt(((length(g1)-1)*var(g1) + (length(g2)-1)*var(g2)) / (length(g1)+length(g2)-2))
    d <- (mean(g2) - mean(g1)) / ps
    cat(sprintf("  %s vs %s: d = %.4f\n", dir_levels[i], dir_levels[j], d))
  }
}
cat("\n")

# Cohen's d for rf condition
cat("Cohen's d within rigid-flexible condition (r->f - f->r):\n")
r_to_f <- data_rf$te_value[data_rf$direction_type == "r_to_f"]
f_to_r <- data_rf$te_value[data_rf$direction_type == "f_to_r"]
ps_rf <- sqrt(((length(r_to_f)-1)*var(r_to_f) + (length(f_to_r)-1)*var(f_to_r)) / 
              (length(r_to_f)+length(f_to_r)-2))
d_rf <- (mean(r_to_f) - mean(f_to_r)) / ps_rf
cat(sprintf("  d = %.4f\n\n", d_rf))

# Supplementary t-tests
cat("7. SUPPLEMENTARY T-TESTS\n\n")

cat("Welch's t-test for target type (all data):\n")
ttest_target <- t.test(te_value ~ target_type, data = data_conv)
cat(sprintf("  t = %.4f, df = %.2f, p = %.6f\n", 
            ttest_target$statistic, ttest_target$parameter, ttest_target$p.value))
cat(sprintf("  Mean rigid target: %.4f\n", ttest_target$estimate[1]))
cat(sprintf("  Mean flexible target: %.4f\n", ttest_target$estimate[2]))
cat("\n")

cat("Welch's t-test within rigid-flexible condition (r->f vs f->r):\n")
ttest_rf <- t.test(te_value ~ direction_type, data = data_rf)
cat(sprintf("  t = %.4f, df = %.2f, p = %.6f\n",
            ttest_rf$statistic, ttest_rf$parameter, ttest_rf$p.value))
cat("\n")

cat("8. SUMMARY\n\n")
cat("Main hypothesis (flexible targets > rigid targets): ")
if (anova(model_target)$`Pr(>F)` < 0.05 && 
    summary(model_target)$coefficients["target_typeflexible", "Estimate"] > 0) {
  cat("SUPPORTED\n")
} else {
  cat("NOT SUPPORTED\n")
}
cat(sprintf("  Target type effect: F = %.2f, p = %.4f\n", 
            anova(model_target)$`F value`, anova(model_target)$`Pr(>F)`))
cat(sprintf("  Effect size (Cohen's d): %.4f\n", cohens_d_target))
cat("\n")

cat("Source x Target interaction:\n")
int_anova <- anova(model_full)
cat(sprintf("  F = %.2f, p = %.4f\n", 
            int_anova["source_type:target_type", "F value"],
            int_anova["source_type:target_type", "Pr(>F)"]))
cat("\n")

sink()

# Generate figure
pdf("results/ste_target_effect.pdf", width = 4, height = 4)

# Calculate means and SEM by target type
stats_target <- aggregate(te_value ~ target_type, data = data_conv,
                          FUN = function(x) c(mean = mean(x), 
                                              se = sd(x)/sqrt(length(x))))
stats_target <- data.frame(target_type = stats_target$target_type,
                           mean = stats_target$te_value[, "mean"],
                           se = stats_target$te_value[, "se"])

# Set up plot
par(mar = c(4, 4, 2, 1))
bp <- barplot(stats_target$mean, 
              names.arg = c("Rigid", "Flexible"),
              ylim = c(0, 0.6),
              col = c("grey60", "grey30"),
              border = NA,
              ylab = "Semantic Transfer Entropy",
              xlab = "Target Cognitive Style",
              main = "")

# Add error bars
arrows(bp, stats_target$mean - stats_target$se,
       bp, stats_target$mean + stats_target$se,
       angle = 90, code = 3, length = 0.05, lwd = 1.5)

# Add significance bracket
lines(c(bp[1], bp[1], bp[2], bp[2]), c(0.54, 0.56, 0.56, 0.54), lwd = 1)
text(mean(bp), 0.58, "*", cex = 1.5)

dev.off()
