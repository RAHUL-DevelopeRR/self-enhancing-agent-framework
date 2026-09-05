# Specialist Training Path

Do not pretrain the first 300M-class workers from random initialization. Begin
with a compact pretrained base, then use domain-adaptive continued pretraining,
teacher distillation and supervised fine-tuning. Runtime independence does not
require wasting the general language competence already present in a base model.

Recommended sequence:

1. Run V0 and collect `.cntx` outcomes.
2. Export only feedback-followed-by-success examples with
   `python training/export_examples.py`.
3. Add curated Shell/FileOps cases, adversarial ambiguity cases and negative
   safety examples.
4. Split by repository/task family to prevent leakage.
5. Fine-tune Shell and FileOps separately using LoRA/QLoRA.
6. Merge or load the accepted adapter, convert it to GGUF, and create a new
   versioned worker ID.
7. Promote only if its frozen capability benchmark beats the previous worker.

The minimum evaluation suite must cover schema validity, task correctness,
Windows/POSIX path quoting, destructive-action rejection, out-of-scope detection,
missing-context behavior and recovery from actual stderr/tool feedback.
