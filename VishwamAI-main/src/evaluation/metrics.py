def build_metrics(args):
    # Build metrics based on the arguments
    metrics = []
    if args.bleu:
        metrics.append(Bleu(4))
    if args.chrf:
        metrics.append(Chrf())
    if args.rouge:
        metrics.append(Rouge())
    if args.bertscore:
        metrics.append(Bertscore())
    return metrics
