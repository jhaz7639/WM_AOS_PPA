# =============================================================================
# Longitudinal speech measures in nfvPPA (PNFA) vs lvPPA (Logopenic)
# Linear mixed models: diagnosis x time, random intercept for participant (PID)
#
# Reproduces the corrected SPSS model:
#   MIXED <measure> BY SPSS_Dx WITH DaysMRI
#     /FIXED = SPSS_Dx DaysMRI SPSS_Dx*DaysMRI | SSTYPE(3)
#     /METHOD = REML
#     /EMMEANS = TABLES(SPSS_Dx) WITH (DaysMRI = 0 ... 1825) COMPARE(SPSS_Dx) ADJ(SIDAK)
#     /RANDOM = INTERCEPT | SUBJECT(PID) COVTYPE(UN)
#
# Outputs (CSV, saved to the output folder set below):
#   lme_type3_tests.csv        Type III F-tests (+ partial eta squared)
#   lme_fixed_effects.csv      parameter estimates
#   lme_covariance_params.csv  random intercept + residual variance, ICC
#   lme_emmeans.csv            estimated marginal means at 0, 1, 2 years
#   lme_emmeans_smooth.csv     EMMs on a fine grid (for plotting the curve + CI band)
#   lme_pairwise.csv           PNFA - Logopenic at each time point (Sidak) + Cohen's d
#   lme_slopes.csv             annual rate of change per group, and slope difference
#   lme_r2.csv                 marginal / conditional R squared
#   lme_random_slopes.csv      sensitivity check: does adding random slopes help?
# =============================================================================

# ---- One-time install (run once in the Console, not every time) -------------
# install.packages(c("readxl", "lme4", "lmerTest", "emmeans"))

library(readxl)
library(lmerTest)
library(emmeans)
emm_options(lmer.df = "satterthwaite")   # SPSS uses Satterthwaite df


# ---- 1. Paths ----------------------------------------------------------------
data_file  <- #/path/to/your/data.xlsx"  # Excel file with the individual level speech measures and covariates
output_dir <- #path/to/your/output/folder"  # where to save the CSV outputs


# ---- 2. Settings ---------------------------------------------------------------
dx_groups    <- c("PNFA", "Logopenic")   # exact labels in SummaryPrimaryDiagnosis
dx_reference <- "Logopenic"              # SPSS reference (SPSS_Dx = 2) -> matches SPSS tables
measures     <- c("SWPVI", "SWduration", "WSPVI", "WSduration")
emm_days     <- c(0, 365, 730, 1095)  
days_per_year <- 365.25


# ---- 3. Load and prepare data --------------------------------------------------
dat <- as.data.frame(read_excel(data_file))
names(dat) <- trimws(names(dat))

dat <- subset(dat, SummaryPrimaryDiagnosis %in% dx_groups)
dat$SPSS_Dx <- relevel(factor(dat$SummaryPrimaryDiagnosis, levels = dx_groups),
                       ref = dx_reference)
dat$PID     <- factor(dat$PID)              # participant ID: same across visits
dat$DaysMRI <- suppressWarnings(as.numeric(dat$DaysMRI))
dat$Years   <- dat$DaysMRI / days_per_year  # used only for the random-slope check
for (m in measures) dat[[m]] <- suppressWarnings(as.numeric(dat[[m]]))

cat("Rows per diagnosis:\n");            print(table(dat$SPSS_Dx))
cat("\nParticipants by number of visits:\n"); print(table(table(droplevels(dat$PID))))


# ---- 4. Fit models -------------------------------------------------------------
out <- list(type3 = list(), fixed = list(), cov = list(), emm = list(), emm_smooth = list(),
            pairwise = list(), slopes = list(), r2 = list(), rslopes = list())

for (m in measures) {

  # Listwise deletion on the model variables (as SPSS does)
  d <- droplevels(dat[!is.na(dat[[m]]) & !is.na(dat$DaysMRI) & !is.na(dat$SPSS_Dx), ])

  fit <- lmer(as.formula(paste(m, "~ SPSS_Dx * DaysMRI + (1 | PID)")),
              data = d, REML = TRUE)

  # -- Variance components ------------------------------------------------------
  vc      <- as.data.frame(VarCorr(fit))
  var_int <- vc$vcov[vc$grp == "PID"]
  var_res <- vc$vcov[vc$grp == "Residual"]
  sd_total <- sqrt(var_int + var_res)       # SD of a single observation (for Cohen's d)
  out$cov[[m]] <- data.frame(measure = m,
                             component = c("Participant intercept", "Residual"),
                             variance = c(var_int, var_res),
                             SD = sqrt(c(var_int, var_res)),
                             ICC = c(var_int / (var_int + var_res), NA))

  # -- Type III tests + partial eta squared -------------------------------------
  t3 <- as.data.frame(anova(fit, type = 3, ddf = "Satterthwaite"))
  t3$partial_eta2 <- with(t3, (`F value` * NumDF) / (`F value` * NumDF + DenDF))
  out$type3[[m]] <- cbind(measure = m, effect = rownames(t3), t3)

  # -- Parameter estimates --------------------------------------------------------
  fx <- as.data.frame(coef(summary(fit)))
  fx$ci_lower <- fx$Estimate - qt(0.975, fx$df) * fx$`Std. Error`
  fx$ci_upper <- fx$Estimate + qt(0.975, fx$df) * fx$`Std. Error`
  out$fixed[[m]] <- cbind(measure = m, term = rownames(fx), fx)

  # -- Estimated marginal means --------------------------------------------------
  em <- emmeans(fit, ~ SPSS_Dx | DaysMRI, at = list(DaysMRI = emm_days))
  out$emm[[m]] <- cbind(measure = m, as.data.frame(em))

  # Fine grid across the observed follow-up range, for plotting
  em_s <- emmeans(fit, ~ SPSS_Dx | DaysMRI,
                  at = list(DaysMRI = seq(0, max(d$DaysMRI), length.out = 150)))
  out$emm_smooth[[m]] <- cbind(measure = m, as.data.frame(em_s))

  # -- Post hoc: PNFA - Logopenic at each time point ------------------------------
  # reverse = TRUE gives PNFA - Logopenic, the same direction as SPSS (I = 1, J = 2)
  pw <- pairs(em, reverse = TRUE)

  # (a) Exactly as SPSS: Sidak applied within each time point.
  #     With only 2 groups there is 1 comparison per time point, so this
  #     is identical to the unadjusted p-value.
  pw_spss <- as.data.frame(summary(pw, infer = c(TRUE, TRUE), adjust = "sidak"))

  # (b) Sidak across all time points in emm_days (controls error over the family)
  pw_fam <- as.data.frame(summary(pw, by = NULL, infer = c(TRUE, TRUE), adjust = "sidak"))

  pw_out <- data.frame(
    measure    = m,
    contrast   = pw_spss$contrast,
    DaysMRI    = pw_spss$DaysMRI,
    Years      = pw_spss$DaysMRI / days_per_year,
    difference = pw_spss$estimate,
    SE         = pw_spss$SE,
    df         = pw_spss$df,
    t          = pw_spss$t.ratio,
    ci_lower   = pw_spss$lower.CL,
    ci_upper   = pw_spss$upper.CL,
    p_sidak_per_timepoint   = pw_spss$p.value,   # = SPSS "Sig."
    ci_lower_sidak_across   = pw_fam$lower.CL[match(pw_spss$DaysMRI, pw_fam$DaysMRI)],
    ci_upper_sidak_across   = pw_fam$upper.CL[match(pw_spss$DaysMRI, pw_fam$DaysMRI)],
    p_sidak_across          = pw_fam$p.value[match(pw_spss$DaysMRI, pw_fam$DaysMRI)],
    cohens_d   = pw_spss$estimate / sd_total,
    d_ci_lower = pw_spss$lower.CL / sd_total,    # approximate: ignores uncertainty in the SD
    d_ci_upper = pw_spss$upper.CL / sd_total
  )
  out$pairwise[[m]] <- pw_out

  # -- Annual rate of change in each group, and the difference --------------------
  tr   <- emtrends(fit, ~ SPSS_Dx, var = "DaysMRI")
  tr_g <- as.data.frame(summary(tr, infer = c(TRUE, TRUE)))
  tr_d <- as.data.frame(summary(pairs(tr, reverse = TRUE), infer = c(TRUE, TRUE)))

  sl <- data.frame(
    measure  = m,
    group    = c(as.character(tr_g$SPSS_Dx), as.character(tr_d$contrast)),
    slope_per_year = c(tr_g$DaysMRI.trend, tr_d$estimate) * days_per_year,
    SE       = c(tr_g$SE, tr_d$SE) * days_per_year,
    df       = c(tr_g$df, tr_d$df),
    ci_lower = c(tr_g$lower.CL, tr_d$lower.CL) * days_per_year,
    ci_upper = c(tr_g$upper.CL, tr_d$upper.CL) * days_per_year,
    t        = c(tr_g$t.ratio, tr_d$t.ratio),
    p        = c(tr_g$p.value, tr_d$p.value)
  )
  # Standardised annual change: SD units per year
  sl$d_per_year          <- sl$slope_per_year / sd_total
  sl$d_per_year_ci_lower <- sl$ci_lower / sd_total
  sl$d_per_year_ci_upper <- sl$ci_upper / sd_total
  out$slopes[[m]] <- sl

  # -- Marginal / conditional R squared (Nakagawa & Schielzeth) -----------------
  var_fixed <- var(as.vector(model.matrix(fit) %*% fixef(fit)))
  out$r2[[m]] <- data.frame(
    measure = m,
    R2_marginal    = var_fixed / (var_fixed + var_int + var_res),
    R2_conditional = (var_fixed + var_int) / (var_fixed + var_int + var_res),
    n_obs          = nrow(d),
    n_participants = nlevels(d$PID)
  )

  # -- Sensitivity: random intercept vs random intercept + slope -----------------
  # Time in years here so the slope variance is on a sensible scale.
  f0 <- as.formula(paste(m, "~ SPSS_Dx * Years + (1 | PID)"))
  f1 <- as.formula(paste(m, "~ SPSS_Dx * Years + (1 + Years | PID)"))
  m0 <- lmer(f0, data = d, REML = TRUE)
  m1 <- tryCatch(
    lmer(f1, data = d, REML = TRUE,
         control = lmerControl(check.nobs.vs.nRE = "ignore")),
    error = function(e) NULL)

  if (is.null(m1)) {
    out$rslopes[[m]] <- data.frame(measure = m, fitted = FALSE, singular = NA,
                                   slope_SD = NA, LRT_chisq = NA, LRT_p = NA,
                                   interaction_F = NA, interaction_df = NA, interaction_p = NA)
  } else {
    lrt  <- anova(m0, m1, refit = FALSE)          # same fixed effects, REML fits
    chi  <- lrt$Chisq[2]
    # Variance tested on its boundary: p from a 50:50 mix of chi-sq(1) and chi-sq(2)
    p_mix <- 0.5 * pchisq(chi, 1, lower.tail = FALSE) + 0.5 * pchisq(chi, 2, lower.tail = FALSE)
    a1 <- as.data.frame(anova(m1, type = 3, ddf = "Satterthwaite"))
    vc1 <- as.data.frame(VarCorr(m1))
    out$rslopes[[m]] <- data.frame(
      measure = m, fitted = TRUE, singular = isSingular(m1),
      slope_SD = vc1$sdcor[vc1$grp == "PID" & vc1$var1 == "Years" & is.na(vc1$var2)],
      LRT_chisq = chi, LRT_p = p_mix,
      interaction_F  = a1["SPSS_Dx:Years", "F value"],
      interaction_df = a1["SPSS_Dx:Years", "DenDF"],
      interaction_p  = a1["SPSS_Dx:Years", "Pr(>F)"])
  }

  cat("\n==================", m, "==================\n")
  cat("N observations:", nrow(d), " | N participants:", nlevels(d$PID), "\n\n")
  print(t3)
}

tabs <- lapply(out, function(x) { y <- do.call(rbind, x); rownames(y) <- NULL; y })


# ---- 5. Key results in the Console ---------------------------------------------
num <- function(df) { df[] <- lapply(df, function(z) if (is.numeric(z)) signif(z, 4) else z); df }

cat("\n\n===== Diagnosis x time interaction =====\n")
print(num(subset(tabs$type3, effect == "SPSS_Dx:DaysMRI")), row.names = FALSE)

cat("\n===== Annual rate of change (units per year) =====\n")
print(num(tabs$slopes), row.names = FALSE)

cat("\n===== PNFA - Logopenic at each time point =====\n")
print(num(tabs$pairwise[, c("measure", "Years", "difference", "ci_lower", "ci_upper",
                            "p_sidak_per_timepoint", "p_sidak_across", "cohens_d")]),
      row.names = FALSE)

cat("\n===== R squared =====\n")
print(num(tabs$r2), row.names = FALSE)

cat("\n===== Random-slope sensitivity check =====\n")
print(num(tabs$rslopes), row.names = FALSE)


# ---- 6. Save --------------------------------------------------------------------
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
files <- c(type3 = "lme_type3_tests.csv", fixed = "lme_fixed_effects.csv",
           cov = "lme_covariance_params.csv", emm = "lme_emmeans.csv",
           emm_smooth = "lme_emmeans_smooth.csv",
           pairwise = "lme_pairwise.csv", slopes = "lme_slopes.csv",
           r2 = "lme_r2.csv", rslopes = "lme_random_slopes.csv")
for (k in names(files)) write.csv(tabs[[k]], file.path(output_dir, files[[k]]), row.names = FALSE)
cat("\nSaved", length(files), "CSV files to:", output_dir, "\n")
