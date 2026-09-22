#!/usr/bin/env python3
"""Fast BLiMP evaluation with batched inference."""
import argparse, json, os, sys, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

BLIMP_SUBTASKS = [
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
    "drop_argument", "ellipsis_n_bar_1", "ellipsis_n_bar_2",
    "existential_there_object_raising", "existential_there_quantifiers_1",
    "existential_there_quantifiers_2", "existential_there_subject_raising",
    "expletive_it_object_raising", "inchoative", "intransitive",
    "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question", "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2",
    "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
    "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1", "superlative_quantifiers_2",
    "tough_vs_raising_1", "tough_vs_raising_2",
    "transitive", "wh_island",
    "wh_questions_object_gap", "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]

BLIMP_CATEGORIES = {
    "D-N AGR": ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"],
    "S-V AGR": ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"],
    "ANA. AGR": ["anaphor_gender_agreement", "anaphor_number_agreement"],
    "ARG. STR": ["animate_subject_passive", "animate_subject_trans", "causative",
        "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2", "transitive"],
    "BINDING": ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
        "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3", "principle_A_reconstruction"],
    "CASE": ["tough_vs_raising_1", "tough_vs_raising_2"],
    "ELLIPSIS": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
    "FILLER-GAP": ["wh_questions_object_gap", "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance"],
    "IRREGULAR": ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"],
    "ISLAND": ["adjunct_island", "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question", "left_branch_island_simple_question",
        "sentential_subject_island", "wh_island"],
    "QUANTIFIERS": ["existential_there_object_raising", "existential_there_quantifiers_1",
        "existential_there_quantifiers_2", "existential_there_subject_raising",
        "expletive_it_object_raising", "superlative_quantifiers_1", "superlative_quantifiers_2"],
    "NPI": ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
        "only_npi_licensor_present", "only_npi_scope",
        "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope"],
}


def batch_logprobs(model, tokenizer, sentences, device, batch_size=32):
    """Compute log probabilities for a list of sentences using batching."""
    all_logprobs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs * shift_mask  # mask padding
            
            sent_logprobs = token_log_probs.sum(dim=-1)  # sum per sentence
            all_logprobs.extend(sent_logprobs.cpu().tolist())
    
    return all_logprobs


def load_model(model_path, model_type, device):
    if model_type == "dynamic_alibi":
        from forgetting_transformer.model.dynamic_alibi.modeling_dynamic_alibi import DynamicAlibiForCausalLM
        model = DynamicAlibiForCausalLM.from_pretrained(model_path)
    elif model_type in ("dynamic_forgetting", "forgetting_gate"):
        from forgetting_transformer.model.dynamic_forgetting.modeling_dynamic_forgetting import DynamicForgettingForCausalLM
        model = DynamicForgettingForCausalLM.from_pretrained(model_path)
    elif model_type == "dynamic_sliding_window":
        from forgetting_transformer.model.dynamic_sliding_window.modeling_dynamic_sliding_window import DynamicSlidingWindowForCausalLM
        model = DynamicSlidingWindowForCausalLM.from_pretrained(model_path)
    elif model_type == "forgetting_transformer":
        from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import ForgettingTransformerForCausalLM
        model = ForgettingTransformerForCausalLM.from_pretrained(model_path)
    elif model_type == "alibi":
        from forgetting_transformer.model.alibi.modeling_alibi import AlibiForCausalLM
        model = AlibiForCausalLM.from_pretrained(model_path)
    elif model_type == "transformer":
        from forgetting_transformer.model.transformer.modeling_transformer import TransformerForCausalLM
        model = TransformerForCausalLM.from_pretrained(model_path)
    elif model_type == "sliding_window":
        from forgetting_transformer.model.sliding_window.modeling_sliding_window import SlidingWindowForCausalLM
        model = SlidingWindowForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model = model.to(device).eval()
    return model


def evaluate_blimp(model, tokenizer, device, batch_size=64):
    results = {}
    for subtask in BLIMP_SUBTASKS:
        try:
            ds = load_dataset("blimp", subtask, split="train")
        except Exception as e:
            print(f"  Skip {subtask}: {e}")
            continue
        
        good_sents = [ex["sentence_good"] for ex in ds]
        bad_sents = [ex["sentence_bad"] for ex in ds]
        
        good_lps = batch_logprobs(model, tokenizer, good_sents, device, batch_size)
        bad_lps = batch_logprobs(model, tokenizer, bad_sents, device, batch_size)
        
        correct = sum(1 for g, b in zip(good_lps, bad_lps) if g > b)
        total = len(good_lps)
        accuracy = correct / total * 100
        results[subtask] = {"accuracy": accuracy, "correct": correct, "total": total}
        print(f"  {subtask}: {accuracy:.1f}%", flush=True)
    
    category_results = {}
    for cat, subs in BLIMP_CATEGORIES.items():
        c = sum(results[s]["correct"] for s in subs if s in results)
        t = sum(results[s]["total"] for s in subs if s in results)
        category_results[cat] = c / t * 100 if t > 0 else 0
    
    all_c = sum(r["correct"] for r in results.values())
    all_t = sum(r["total"] for r in results.values())
    overall = all_c / all_t * 100 if all_t > 0 else 0
    
    return results, category_results, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading {args.model}...", flush=True)
    model = load_model(args.model, args.model_type, args.device)
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    
    results, cats, overall = evaluate_blimp(model, tokenizer, args.device, args.batch_size)
    
    print(f"\n{'='*50}")
    for c, a in sorted(cats.items()):
        print(f"  {c:15s}: {a:.1f}%")
    print(f"  {'OVERALL':15s}: {overall:.1f}%")
    print(f"{'='*50}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "model_type": args.model_type,
                       "overall": overall, "categories": cats,
                       "subtasks": {k: v["accuracy"] for k, v in results.items()}}, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
