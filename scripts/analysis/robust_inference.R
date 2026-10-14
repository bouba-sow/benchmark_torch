# Robust inference for the matched-scale constraint regressions (reviewer step A).
#
# Input: <dir>/rows_{embedding,surprisal}.csv from scripts/analysis/robust_inference_export.py.
# Usage: Rscript scripts/analysis/robust_inference.R <dir> [n_boot=999] [n_cores=8]
#
# For each track it fits, all with REML and Satterthwaite df (lmerTest):
#   1. Specification ladder, separate behavioral / neural fits (the paper's Table 2 is "S0"):
#        S0  y ~ M*D + size + ppl + (1|unit)
#        S1  S0 + (1|lineage) + (1|model) [+ (1|corpus), behavioral]       dependence between models / corpus units
#        S2  S1 + (0+size|unit) + (0+ppl|unit)                              unit-specific slopes   <- preferred
#   2. Joint model with domain interactions (tests the behavioral-neural dissociation directly):
#        y ~ (M*D + size + ppl) * domain + (1|unit) + (0+size|unit) + (0+ppl|unit)
#            + (1|lineage) + (1|lineage:domain) + (1|model) + (1|model:domain) + (1|corpus)
#      fit on the ceiling-normalized score and on the within-unit robust z (domains have very different scales).
#   3. Robustness of S2 (separate fits): all units / drop dagger units / winsorize 5-95% within unit / robust z within
#      unit / drop GPT-BERT / + log10 depth / leave-one-unit-out.
#   4. Model ladder (S2 random effects): constraints only / + size / + ppl / + both.
#   5. Lineage cluster bootstrap of S0 (clusters resampled with replacement within each constraint group), percentile CI.
#   6. Per-unit OLS coefficients (y ~ M*D + size + ppl within each unit).
# size = log10 params, ppl = log10 perplexity, both centered. "both" = M + D + M:D (combined Constrained-BabyLM contrast).
#
# Outputs in <dir>: coefficients.csv, per_unit.csv, bootstrap.csv, forest_<track>.pdf, per_unit_<track>.pdf,
# summary.txt (fit warnings and sample sizes).

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(data.table); library(ggplot2); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "reports/robust_inference"
n_boot <- if (length(args) >= 2) as.integer(args[2]) else 999L
n_cores <- if (length(args) >= 3) as.integer(args[3]) else 8L
RNGkind("L'Ecuyer-CMRG"); set.seed(20261014)  # reproducible streams across mclapply workers
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
log_lines <- character()
note <- function(...) { s <- paste0(...); log_lines <<- c(log_lines, s); message(s) }

TERMS <- c("M", "D", "M:D", "size", "ppl")
LABEL <- c(M = "Memory only", D = "Data only", "M:D" = "Memory x data", both = "Both constraints",
           size = "log10 parameters", ppl = "log10 perplexity")

prep <- function(d) {
  d <- as.data.table(d)
  d[, `:=`(unit = dataset, model = model_id, M = memory, D = data,
           size = log_params - mean(log_params), ppl = log_ppl - mean(log_ppl),
           depth = log10(n_layers) - mean(log10(n_layers)), y = score_norm)]
  d[, domain := factor(modality, levels = c("behavioral", "neural"))]
  d[]
}
robust_z <- function(x) { s <- IQR(x) / 1.349; if (!(s > 0)) s <- mad(x); (x - median(x)) / s }
winsor <- function(x, p = .05) { q <- quantile(x, c(p, 1 - p)); pmin(pmax(x, q[1]), q[2]) }

re_terms <- function(spec, d) {
  r <- "(1|unit)"
  if (spec %in% c("S1", "S2")) {
    r <- c(r, "(1|lineage)", "(1|model)")
    if (uniqueN(d$corpus) < uniqueN(d$unit)) r <- c(r, "(1|corpus)")
  }
  if (spec == "S2") r <- c(r, "(0+size|unit)", "(0+ppl|unit)")
  paste(r, collapse = " + ")
}

fit <- function(d, fixed, spec) {
  f <- as.formula(paste("y ~", fixed, "+", re_terms(spec, d)))
  w <- character()
  m <- withCallingHandlers(lmer(f, data = d, REML = TRUE, control = ctrl),
                           warning = function(e) { w <<- c(w, conditionMessage(e)); invokeRestart("muffleWarning") })
  list(m = m, warn = paste(unique(c(w, if (isSingular(m)) "singular")), collapse = "; "))
}

# Fixed-effect estimates for the named terms plus the combined contrast M + D + M:D.
coefs <- function(m, terms = TERMS, suffix = "") {
  fe <- fixef(m); out <- list()
  for (t in terms) {
    nm <- paste0(t, suffix); if (!nm %in% names(fe)) next
    L <- setNames(as.numeric(names(fe) == nm), names(fe))
    out[[t]] <- contest1D(m, L)
  }
  if (all(paste0(c("M", "D", "M:D"), suffix) %in% names(fe))) {
    L <- setNames(as.numeric(names(fe) %in% paste0(c("M", "D", "M:D"), suffix)), names(fe))
    out[["both"]] <- contest1D(m, L)
  }
  if (!length(out)) stop("no requested terms among fixed effects: ", paste(names(fe), collapse = ", "))
  r <- rbindlist(lapply(names(out), function(t) {
    x <- out[[t]]; data.table(term = t, est = x$Estimate, se = x$`Std. Error`, df = x$df, p = x$`Pr(>|t|)`)
  }))
  r[, `:=`(lo = est - qt(.975, df) * se, hi = est + qt(.975, df) * se)]
  r[]
}

all_rows <- list(); unit_rows <- list(); boot_rows <- list()
add <- function(r, ...) { if (!is.null(r) && nrow(r)) all_rows[[length(all_rows) + 1]] <<- cbind(data.table(...), r) }
FIXED <- "M*D + size + ppl"

for (track in c("embedding", "surprisal")) {
  d0 <- prep(fread(file.path(dir, sprintf("rows_%s.csv", track))))
  d0[, yz := robust_z(y), by = unit]
  d0[, yw := winsor(y), by = unit]
  note(sprintf("[%s] rows %d (beh %d / neu %d), models %d, lineages %d", track, nrow(d0),
               sum(d0$domain == "behavioral"), sum(d0$domain == "neural"), uniqueN(d0$model), uniqueN(d0$lineage)))

  for (dom in levels(d0$domain)) {
    d <- d0[domain == dom]
    # 1. specification ladder
    for (spec in c("S0", "S1", "S2")) {
      f <- fit(d, FIXED, spec)
      add(coefs(f$m), track = track, domain = dom, analysis = "ladder", variant = spec, warn = f$warn)
    }
    # 3. robustness of S2
    variants <- list(
      all = list(d = d, fixed = FIXED),
      no_dagger = list(d = d[dagger == 0], fixed = FIXED),
      winsor = list(d = copy(d)[, y := yw], fixed = FIXED),
      robust_z = list(d = copy(d)[, y := yz], fixed = FIXED),
      no_gptbert = list(d = d[gptbert == 0], fixed = FIXED),
      depth = list(d = d, fixed = paste(FIXED, "+ depth")))
    for (v in names(variants)) {
      if (v == "no_dagger" && !any(d$dagger == 1)) next
      vv <- variants[[v]]
      f <- tryCatch(fit(vv$d, vv$fixed, "S2"), error = function(e) { note("FAIL ", track, dom, v, ": ", conditionMessage(e)); NULL })
      if (!is.null(f)) add(coefs(f$m, c(TERMS, "depth")), track = track, domain = dom, analysis = "robustness",
                           variant = v, warn = f$warn)
    }
    for (u in unique(d$unit)) {
      f <- tryCatch(fit(d[unit != u], FIXED, "S2"), error = function(e) NULL)
      if (!is.null(f)) add(coefs(f$m), track = track, domain = dom, analysis = "leave_one_unit_out",
                           variant = u, warn = f$warn)
    }
    # 4. model ladder (direct vs total associations of the constraints)
    for (fx in c("M*D", "M*D + size", "M*D + ppl", FIXED)) {
      f <- fit(d, fx, "S2")
      add(coefs(f$m), track = track, domain = dom, analysis = "covariate_ladder", variant = fx, warn = f$warn)
    }
    # 5. lineage cluster bootstrap of S0, stratified by constraint group
    cl <- unique(d[, .(lineage, group)])
    one <- function(b) {
      pick <- cl[, .(lineage = sample(lineage, .N, replace = TRUE)), by = group]
      bd <- rbindlist(lapply(seq_len(nrow(pick)), function(i) {
        x <- d[lineage == pick$lineage[i]]; x[, model := paste0(model, "#", i)]; x }))
      m <- tryCatch(suppressWarnings(lmer(y ~ M*D + size + ppl + (1|unit), data = bd, REML = TRUE, control = ctrl)),
                    error = function(e) NULL)
      if (is.null(m)) return(NULL)
      fe <- fixef(m); fe <- c(fe[intersect(TERMS, names(fe))], both = sum(fe[c("M", "D", "M:D")]))
      data.table(b = b, term = names(fe), est = unname(fe))
    }
    bt <- rbindlist(mclapply(seq_len(n_boot), one, mc.cores = n_cores))
    boot_rows[[length(boot_rows) + 1]] <- cbind(data.table(track = track, domain = dom), bt)
    s0 <- coefs(fit(d, FIXED, "S0")$m)
    bs <- bt[, .(lo = quantile(est, .025, na.rm = TRUE), hi = quantile(est, .975, na.rm = TRUE),
                 se = sd(est, na.rm = TRUE), p = 2 * min(mean(est <= 0, na.rm = TRUE), mean(est >= 0, na.rm = TRUE)),
                 n_ok = sum(!is.na(est))), by = term]
    bs <- merge(s0[, .(term, est)], bs, by = "term")
    bs[, df := NA_real_]
    add(bs[, .(term, est, se, df, p, lo, hi)], track = track, domain = dom, analysis = "lineage_bootstrap",
        variant = sprintf("S0, %d resamples", n_boot), warn = sprintf("%d/%d resamples ok", min(bs$n_ok), n_boot))
    # 6. per-unit OLS
    for (u in unique(d$unit)) {
      m <- lm(y ~ M*D + size + ppl, data = d[unit == u]); s <- summary(m)$coefficients
      ci <- confint(m)
      keep <- intersect(TERMS, rownames(s))
      unit_rows[[length(unit_rows) + 1]] <- data.table(track = track, domain = dom, unit = u, term = keep,
        est = s[keep, 1], se = s[keep, 2], p = s[keep, 4], lo = ci[keep, 1], hi = ci[keep, 2],
        dagger = any(d[unit == u]$dagger == 1))
    }
  }

  # 2. joint model with domain interactions (behavioral = reference, so "X:domainneural" = neural minus behavioral)
  joint_re <- "(1|unit) + (0+size|unit) + (0+ppl|unit) + (1|lineage) + (1|lineage:domain) + (1|model) + (1|model:domain) + (1|corpus)"
  for (outcome in c("y", "yz")) {
    dd <- copy(d0); dd[, y := get(outcome)]
    w <- character()
    m <- withCallingHandlers(lmer(as.formula(paste("y ~ (M*D + size + ppl) * domain +", joint_re)),
                                  data = dd, REML = TRUE, control = ctrl),
                             warning = function(e) { w <<- c(w, conditionMessage(e)); invokeRestart("muffleWarning") })
    lab <- if (outcome == "y") "ceiling-normalized" else "robust z"
    wtxt <- paste(unique(c(w, if (isSingular(m)) "singular")), collapse = "; ")
    add(coefs(m), track = track, domain = "behavioral", analysis = "joint", variant = lab, warn = wtxt)
    add(coefs(m, suffix = ":domainneural"), track = track, domain = "neural - behavioral", analysis = "joint",
        variant = lab, warn = wtxt)
    # neural-domain effects from the same fit: refit with neural as reference
    dd[, domain := relevel(domain, "neural")]
    m2 <- suppressWarnings(lmer(as.formula(paste("y ~ (M*D + size + ppl) * domain +", joint_re)),
                                data = dd, REML = TRUE, control = ctrl))
    add(coefs(m2), track = track, domain = "neural", analysis = "joint", variant = lab, warn = wtxt)
  }
}

co <- rbindlist(all_rows, fill = TRUE)
co[, label := LABEL[term]]
fwrite(co, file.path(dir, "coefficients.csv"))
pu <- rbindlist(unit_rows); fwrite(pu, file.path(dir, "per_unit.csv"))
fwrite(rbindlist(boot_rows), file.path(dir, "bootstrap.csv"))

# Forest plots: constraint contrasts under the main specifications (ceiling-normalized scale only; the robust-z fits
# are on a different scale and stay in coefficients.csv).
SHOW <- c("ladder: S0" = "S0 (Table 2)", "ladder: S2" = "S2 (preferred)",
          "lineage_bootstrap" = "S0, lineage bootstrap", "joint: ceiling-normalized" = "joint model",
          "robustness: no_dagger" = "S2 without flagged units", "robustness: winsor" = "S2 winsorized")
for (track in unique(co$track)) {
  x <- co[track == get("track", envir = globalenv()) & term %in% c("M", "D", "M:D", "both", "size", "ppl") &
            domain %in% c("behavioral", "neural")]
  x[, spec := ifelse(analysis == "lineage_bootstrap", "lineage_bootstrap", paste(analysis, variant, sep = ": "))]
  x <- x[spec %in% names(SHOW)]
  x[, spec := factor(SHOW[spec], levels = rev(SHOW))]
  x[, term := factor(LABEL[term], levels = LABEL[c("M", "D", "M:D", "both", "size", "ppl")])]
  g <- ggplot(x, aes(est, spec, colour = spec)) +
    geom_vline(xintercept = 0, colour = "grey50") +
    geom_pointrange(aes(xmin = lo, xmax = hi), size = .25) +
    facet_grid(domain ~ term, scales = "free_x") +
    labs(x = "coefficient, ceiling-normalized score units (95% CI)", y = NULL, title = paste(track, "track")) +
    theme_bw(10) + theme(legend.position = "none", axis.text.x = element_text(size = 7))
  ggsave(file.path(dir, sprintf("forest_%s.pdf", track)), g, width = 11, height = 4.5)

  y <- pu[track == get("track", envir = globalenv()) & term %in% c("M", "D")]
  y[, term := LABEL[term]]
  g <- ggplot(y, aes(est, reorder(unit, est), colour = dagger)) + geom_vline(xintercept = 0, colour = "grey50") +
    geom_pointrange(aes(xmin = lo, xmax = hi), size = .2) + facet_grid(domain ~ term, scales = "free", space = "free_y") +
    scale_colour_manual(values = c(`FALSE` = "black", `TRUE` = "firebrick")) +
    labs(x = "per-unit OLS coefficient (95% CI)", y = NULL, colour = "flagged reference", title = paste(track, "track")) +
    theme_bw(9)
  ggsave(file.path(dir, sprintf("per_unit_%s.pdf", track)), g, width = 10, height = 8)
}

# Compact text summary of the headline contrasts.
hl <- co[term %in% c("M", "D", "M:D", "both") & analysis %in% c("ladder", "joint", "lineage_bootstrap", "robustness")]
fmt <- function(r) sprintf("%-22s %-24s %-6s %8.3f [%7.3f, %7.3f]  p=%.3g  %s", paste(r$analysis, r$variant),
                           r$domain, r$term, r$est, r$lo, r$hi, r$p, r$warn)
writeLines(c(log_lines, "", unlist(lapply(split(hl, by = "track"), function(t)
  c(paste("==", t$track[1]), sapply(seq_len(nrow(t)), function(i) fmt(t[i]))))),
  "", "LOUO ranges (min..max across left-out units):",
  co[analysis == "leave_one_unit_out", .(range = sprintf("%.3f .. %.3f", min(est), max(est)),
                                         sign_flips = sum(sign(est) != sign(median(est)))),
     by = .(track, domain, term)][, sprintf("%s %s %s %s flips=%d", track, domain, term, range, sign_flips)]),
  file.path(dir, "summary.txt"))
message("done: ", dir)
