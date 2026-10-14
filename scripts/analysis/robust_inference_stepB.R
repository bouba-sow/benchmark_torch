# Design-based comparisons for the constraint analysis (reviewer step B). Same inputs as robust_inference.R
# (<dir>/rows_{embedding,surprisal}.csv from robust_inference_export.py).
# Usage: Rscript scripts/analysis/robust_inference_stepB.R <dir>
#
#   1. Data dose-response within architecture (Bu et al. checkpoints: 6 memory-limited architectures x
#      {BabyLM 10M, BabyLM 100M, Pile} x {2, 4 layers} = 36 models, identical code and size within architecture x depth):
#        y ~ data_level + depth4 + (1|arch) + (1|model) + (1|unit) [+ (1|corpus)],  Pile = reference.
#      No size/perplexity covariates: size is fixed within architecture x depth, and perplexity is an outcome of the
#      training data, so adjusting for it would remove part of the data effect.
#   2. FoX matched pairs: Forgetting Transformer vs its own softmax baseline (llama and pro variants, 760M, same data
#      and code). Per unit, the mean over the two pairs of (FoX - baseline); summarized across units of each domain
#      (mean, t-test across units, number of units with FoX ahead).
#   3. Per-mechanism contrasts: the preferred specification (S2 of robust_inference.R) with the single memory dummy
#      replaced by one dummy per mechanism: distance bias (ALiBi, incl. BLOOM), forget gate (Forgetting, FoX),
#      stick-breaking (incl. MoLM/ModuleFormer), hard n-gram window, CBR-RNN; each also interacts with the data factor.
#
# Outputs in <dir>: stepB_coefficients.csv, stepB_fox_pairs.csv, stepB_dose_<track>.pdf, stepB_mechanism_<track>.pdf

suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(data.table); library(ggplot2) })
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "reports/robust_inference"
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

prep <- function(d) {
  d <- as.data.table(d)
  d[, `:=`(unit = dataset, model = model_id, M = memory, D = data, y = score_norm,
           size = log_params - mean(log_params), ppl = log_ppl - mean(log_ppl))]
  d[, domain := factor(modality, levels = c("behavioral", "neural"))]
  bu <- sub("^Lanni-ni/", "", d$model_id)
  d[, bu := grepl("^Lanni-ni/", model_id)]
  d[bu == TRUE, `:=`(arch = sub("_(babylm_10m|babylm_100m|pile)_[24]layer$", "", sub("^Lanni-ni/", "", model_id)),
                     data_level = factor(ifelse(grepl("babylm_10m", model_id), "BabyLM 10M",
                                         ifelse(grepl("babylm_100m", model_id), "BabyLM 100M", "Pile")),
                                         levels = c("Pile", "BabyLM 100M", "BabyLM 10M")),
                     depth4 = as.integer(grepl("4layer", model_id)))]
  d[, mech := fifelse(M == 0, "none",
              fifelse(family %in% c("alibi", "bloom"), "distance bias",
              fifelse(family %in% c("forgetting", "fox"), "forget gate",
              fifelse(family %in% c("stickbreaking", "molm"), "stick-breaking",
              fifelse(grepl("^hard_", family), "hard window",
              fifelse(family == "cbr_rnn", "CBR-RNN", NA_character_))))))]
  stopifnot(!anyNA(d$mech))
  d[, mech := factor(mech, levels = c("none", "distance bias", "forget gate", "stick-breaking", "hard window", "CBR-RNN"))]
  d[]
}

tidy <- function(m, terms) {
  fe <- fixef(m)
  rbindlist(lapply(terms, function(t) {
    if (is.character(t)) { nm <- t; L <- setNames(as.numeric(names(fe) == t), names(fe)) }
    else { nm <- names(t); L <- setNames(as.numeric(names(fe) %in% t[[1]]), names(fe)) }
    if (!any(L != 0) || (is.list(t) && !all(t[[1]] %in% names(fe)))) return(NULL)
    x <- contest1D(m, L)
    data.table(term = nm, est = x$Estimate, se = x$`Std. Error`, df = x$df, p = x$`Pr(>|t|)`)
  }))[, `:=`(lo = est - qt(.975, df) * se, hi = est + qt(.975, df) * se)][]
}
corpus_re <- function(d) if (uniqueN(d$corpus) < uniqueN(d$unit)) " + (1|corpus)" else ""
singular <- function(m) if (isSingular(m)) "singular" else ""

out <- list(); pairs <- list()
for (track in c("embedding", "surprisal")) {
  d0 <- prep(fread(file.path(dir, sprintf("rows_%s.csv", track))))
  for (dom in levels(d0$domain)) {
    d <- d0[domain == dom]

    # 1. dose-response within architecture
    b <- d[bu == TRUE]
    m <- suppressMessages(lmer(as.formula(paste("y ~ data_level + depth4 + (1|arch) + (1|model) + (1|unit)", corpus_re(b))),
                               data = b, REML = TRUE, control = ctrl))
    r <- tidy(m, list("data_levelBabyLM 100M", "data_levelBabyLM 10M", "depth4"))
    r[, term := c("BabyLM 100M - Pile", "BabyLM 10M - Pile", "4 vs 2 layers")]
    m2 <- suppressMessages(lmer(as.formula(paste("y ~ as.numeric(data_level) + depth4 + (1|arch) + (1|model) + (1|unit)",
                                                 corpus_re(b))), data = b, REML = TRUE, control = ctrl))
    r2 <- tidy(m2, list("as.numeric(data_level)")); r2[, term := "linear trend per step (less data)"]
    out[[length(out) + 1]] <- cbind(data.table(track = track, domain = dom, analysis = "dose_response",
                                               n_models = uniqueN(b$model), warn = singular(m)), rbind(r, r2))
    # same, per architecture (descriptive)
    for (a in unique(b$arch)) {
      ma <- suppressMessages(lmer(y ~ data_level + depth4 + (1|unit), data = b[arch == a], REML = TRUE, control = ctrl))
      ra <- tidy(ma, list("data_levelBabyLM 100M", "data_levelBabyLM 10M"))
      ra[, term := paste0(a, ": ", c("BabyLM 100M - Pile", "BabyLM 10M - Pile"))]
      out[[length(out) + 1]] <- cbind(data.table(track = track, domain = dom, analysis = "dose_response_by_arch",
                                                 n_models = uniqueN(b[arch == a]$model), warn = singular(ma)), ra)
    }

    # 2. FoX matched pairs
    f <- d[family == "fox"]
    f[, pair := ifelse(grepl("llama", model_id), "llama", "pro")]
    f[, fox := as.integer(grepl("/fox-", model_id))]
    w <- dcast(f, unit + pair ~ fox, value.var = "y")
    setnames(w, c("0", "1"), c("baseline", "fox"))
    pu <- w[, .(diff = mean(fox - baseline)), by = unit]
    tt <- t.test(pu$diff)
    pairs[[length(pairs) + 1]] <- cbind(data.table(track = track, domain = dom), pu)
    out[[length(out) + 1]] <- data.table(track = track, domain = dom, analysis = "fox_pairs", n_models = 4L,
      warn = sprintf("FoX ahead in %d/%d units", sum(pu$diff > 0), nrow(pu)), term = "FoX - softmax baseline",
      est = mean(pu$diff), se = sd(pu$diff) / sqrt(nrow(pu)), df = nrow(pu) - 1, p = tt$p.value,
      lo = tt$conf.int[1], hi = tt$conf.int[2])

    # 3. per-mechanism contrasts, preferred random-effect structure
    re <- paste0("(1|unit) + (0+size|unit) + (0+ppl|unit) + (1|lineage) + (1|model)", corpus_re(d))
    mm <- suppressMessages(lmer(as.formula(paste("y ~ mech * D + size + ppl +", re)), data = d, REML = TRUE, control = ctrl))
    fe <- names(fixef(mm))
    terms <- list()
    for (k in levels(d$mech)[-1]) {
      main <- paste0("mech", k)
      terms[[length(terms) + 1]] <- setNames(list(main), paste0(k, ": memory only"))
      if (paste0(main, ":D") %in% fe)
        terms[[length(terms) + 1]] <- setNames(list(c(main, "D", paste0(main, ":D"))), paste0(k, ": with BabyLM data"))
    }
    terms[[length(terms) + 1]] <- "D"
    r <- tidy(mm, terms)
    r[term == "D", term := "data only (no memory)"]
    out[[length(out) + 1]] <- cbind(data.table(track = track, domain = dom, analysis = "mechanism",
                                               n_models = uniqueN(d$model), warn = singular(mm)), r)
  }
}
co <- rbindlist(out, fill = TRUE)
fwrite(co, file.path(dir, "stepB_coefficients.csv"))
fwrite(rbindlist(pairs), file.path(dir, "stepB_fox_pairs.csv"))

# Figures: dose-response (model scores by data level, per architecture) and mechanism contrasts.
for (track in c("embedding", "surprisal")) {
  d0 <- prep(fread(file.path(dir, sprintf("rows_%s.csv", track))))
  b <- d0[bu == TRUE, .(y = median(y)), by = .(domain, arch, data_level, depth4, model)]
  b <- b[, .(y = mean(y)), by = .(domain, arch, data_level)]
  g <- ggplot(b, aes(data_level, y, group = arch, colour = arch)) + geom_line() + geom_point() +
    facet_wrap(~domain, scales = "free_y") +
    labs(x = "training data (same architecture and depth)", y = "median normalized score over units\n(mean of 2- and 4-layer models)",
         colour = NULL, title = paste(track, "track")) + theme_bw(10)
  ggsave(file.path(dir, sprintf("stepB_dose_%s.pdf", track)), g, width = 8, height = 3.6)

  x <- co[track == get("track", envir = globalenv()) & analysis == "mechanism"]
  x[, term := factor(term, levels = rev(unique(term)))]
  g <- ggplot(x, aes(est, term)) + geom_vline(xintercept = 0, colour = "grey50") +
    geom_pointrange(aes(xmin = lo, xmax = hi), size = .25) + facet_wrap(~domain, scales = "free_x") +
    labs(x = "contrast with Standard LLMs at matched size and perplexity (95% CI)", y = NULL,
         title = paste(track, "track")) + theme_bw(10)
  ggsave(file.path(dir, sprintf("stepB_mechanism_%s.pdf", track)), g, width = 8, height = 4)
}
message("done: ", dir)
