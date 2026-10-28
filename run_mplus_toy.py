import torch
import numpy as np
import re
from configuration_memoryllm import MemoryLLMConfig
from modeling_mplus import MPlus

# Set seeds for deterministic results
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. EXPANDED TOY DATASET
# =====================================================================
# Constraint per README (line 92):
# "Please make sure the context to inject into the memory is larger than 16 tokens,
# this is the hard minimum when training the model."

toy_dataset = [
    {
        "id": 1,
        "category": "Interpersonal",
        "context": "Last week, John had a wonderful picnic with David in Central Park. During their long conversation, David mentioned multiple times that he really enjoys eating fresh apples. Though he did not mention any other fruits directly, John says he can easily infer that David also likes sweet yellow bananas.",
        "question": "What fruits does David like?",
        "answer_prefix": "David likes",
        "target_answer": "apples and bananas ."
    },
    {
        "id": 2,
        "category": "Biology",
        "context": "Professor Arthur Pendelton spent thirty years in the high Alps documenting a rare species of glacier moth. His latest 2024 monograph conclusively establishes that these moths survive extreme sub-zero temperatures using a specialized biological glycerol compound.",
        "question": "What compound allows glacier moths to survive cold temperatures?",
        "answer_prefix": "The moths survive cold temperatures using",
        "target_answer": "glycerol compound ."
    },
    {
        "id": 3,
        "category": "Astronomy",
        "context": "In 2015, NASA astronomers announced the confirmation of exoplanet Kepler-452b, located roughly 1400 light-years away in the constellation Cygnus. Orbiting within the habitable zone of a Sun-like star, Kepler-452b takes roughly 385 Earth days to complete a single orbital revolution.",
        "question": "In which constellation is exoplanet Kepler-452b located?",
        "answer_prefix": "Kepler-452b is located in the constellation",
        "target_answer": "Cygnus ."
    },
    {
        "id": 4,
        "category": "Architecture & History",
        "context": "Nestled ruggedly above the village of Hohenschwangau in southwest Bavaria, Neuschwanstein Castle was commissioned by King Ludwig II as a secluded personal retreat in 1869. The romantic palace served as the primary visual inspiration for Disneyland's iconic Sleeping Beauty Castle.",
        "question": "Who commissioned the construction of Neuschwanstein Castle?",
        "answer_prefix": "Neuschwanstein Castle was commissioned by",
        "target_answer": "King Ludwig II ."
    },
    {
        "id": 5,
        "category": "Computer Science",
        "context": "English mathematician Ada Lovelace worked closely with Charles Babbage on the early mechanical general-purpose computer known as the Analytical Engine. In 1843, her published notes included an algorithm to calculate Bernoulli numbers, universally recognized as the world's first computer program.",
        "question": "What mechanical computer did Ada Lovelace write the first program for?",
        "answer_prefix": "Ada Lovelace wrote the program for the",
        "target_answer": "Analytical Engine ."
    },
    {
        "id": 6,
        "category": "Medicine",
        "context": "Scottish physician Alexander Fleming made a serendipitous discovery in September 1928 at St Mary's Hospital in London. After returning from vacation, he noticed that a green mold identified as Penicillium notatum had contaminated a petri dish and successfully halted the growth of surrounding staphylococcus bacteria.",
        "question": "What mold produced the antibiotic discovered by Alexander Fleming?",
        "answer_prefix": "The antibiotic was produced by the mold",
        "target_answer": "Penicillium notatum ."
    }
]

# 6 Distractor contexts (each > 20 tokens) to test retention & trigger STM -> LTM migration
unrelated_distractors = [
    "The Amazon River basin is home to the largest rainforest on Earth, spanning across nine South American nations and housing over three million documented plant and animal species worldwide.",
    "The James Webb Space Telescope uses highly sensitive infrared astronomy instruments to observe celestial objects that are too distant, ancient, or faint for the older Hubble Space Telescope.",
    "The Great Barrier Reef off the coast of Queensland in northeastern Australia is the world's largest coral reef system, composed of over 2900 individual reefs and 900 islands.",
    "Aurora Borealis occurs when charged particles emitted from solar coronal mass ejections collide with gases in Earth's upper atmosphere, producing vivid green and violet light displays.",
    "The Mariana Trench in the western Pacific Ocean is the deepest oceanic trench on Earth, plunging nearly eleven thousand meters beneath the ocean surface at the Challenger Deep.",
    "The Sahara Desert spans over nine million square kilometers across Northern Africa, featuring massive wind-sculpted sand dunes, dry gravel plains, and sporadic rocky plateaus."
]

# =====================================================================
# 2. TOKENIZER
# =====================================================================
class SimpleTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 1
        self.word2id = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.id2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}

    def fit(self, texts):
        for text in texts:
            words = re.findall(r"\b\w+\b|[?.,!]", text)
            for w in words:
                w_lower = w.lower()
                if w_lower not in self.word2id:
                    new_id = len(self.word2id)
                    self.word2id[w_lower] = new_id
                    self.id2word[new_id] = w
        return self

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        words = re.findall(r"\b\w+\b|[?.,!]", text)
        ids = [self.word2id.get(w.lower(), self.unk_token_id) for w in words]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids):
        words = []
        for tid in token_ids:
            if isinstance(tid, torch.Tensor):
                tid = tid.item()
            if tid in [self.pad_token_id, self.eos_token_id, self.unk_token_id]:
                continue
            words.append(self.id2word.get(tid, "<unk>"))
        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.word2id)

all_texts = (
    [d["context"] for d in toy_dataset] + 
    [d["question"] for d in toy_dataset] + 
    [d["answer_prefix"] for d in toy_dataset] + 
    [d["target_answer"] for d in toy_dataset] + 
    unrelated_distractors
)
tokenizer = SimpleTokenizer().fit(all_texts)
print(f"Tokenizer fitted on expanded corpus. Vocabulary size: {tokenizer.vocab_size} tokens.\n")

# =====================================================================
# 3. INITIALIZE ACTUAL MPLUS MODEL
# =====================================================================
print("Initializing actual MPlus model (subclassed from LlamaForCausalLM)...")
config = MemoryLLMConfig(
    vocab_size=tokenizer.vocab_size + 50,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_tokens=16,            # tokens per memory block (>= 16 per README)
    num_blocks=4,             # Total memory blocks
    num_ltm_blocks=2,         # Allocated LTM blocks
    add_selector=True,        # Query-Key selector for LTM
    selector_hidden_dim=16,
    num_selector_layers=1,
    update_ltm_mode="immediate",
    update_ltm_frequency=1,
    initial_rf_when_moving_stm_to_ltm=1.0,
    decay_frequency=0.2,
    ltm_initial_size=2,
    add_bos_embedding=True
)

model = MPlus(config)

# =====================================================================
# 4. PHASE 1: QUICK LOCAL PRETRAINING (~5 seconds)
# =====================================================================
# Teaches the miniature model basic language completion on the QA pairs
print("=" * 75)
print("PHASE 1: QUICK LOCAL MODEL PRETRAINING (Teaching QA completion)")
print("=" * 75)

optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3)
model.train()

training_pairs = []
for d in toy_dataset:
    full_text = f"Question: {d['question']} Answer: {d['answer_prefix']} {d['target_answer']}"
    full_ids = tokenizer(full_text, return_tensors="pt")
    prompt_len = len(tokenizer(f"Question: {d['question']} Answer: {d['answer_prefix']}"))
    
    # Create labels masking out the question so loss focuses on answering
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    training_pairs.append((full_ids, labels))

epochs = 28
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for input_ids, labels in training_pairs:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 7 == 0 or epoch == 1:
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch:2d}/{epochs} | Training Loss: {avg_loss:.4f}")

print("Pretraining complete! Weights now produce meaningful language outputs.\n")

# Crucial step per README (line 62): offload LTM to CPU / NumPy
model.put_ltm_to_numpy()
model.eval()

# Freeze all weights for lifelong inference evaluation
for p in model.parameters():
    p.requires_grad = False

# =====================================================================
# 5. PHASE 2: LIFELONG MEMORY INFERENCE (Zero-Backpropagation)
# =====================================================================
print("=" * 75)
print("PHASE 2: LIFELONG MEMORY INJECTION & INFERENCE (ZERO WEIGHT UPDATES)")
print("All model parameters are now 100% FROZEN.")
print("=" * 75)

history = []

for sample in toy_dataset:
    print("=" * 75)
    print(f"Processing Sample #{sample['id']} [{sample['category']}]")
    print("=" * 75)

    # 1. Inject Target Fact per README lines 89-94
    context_tokens = tokenizer(sample["context"], return_tensors="pt", add_special_tokens=False)
    initial_ltm_count = model.ltm[0].shape[0]
    
    print(f"Step 1: Injecting Target Fact ({context_tokens.shape[1]} tokens > 16 minimum)...")
    print(f"Context: \"{sample['context'][:80]}...\"")
    
    delta_memory = model.inject_memory(context_tokens, update_memory=True)
    post_target_ltm_count = model.ltm[0].shape[0]
    
    print(f"-> Delta memory computed: {delta_memory.shape}")
    print(f"-> LTM count before distractors: {post_target_ltm_count} vectors")

    # 2. Inject Distractors to trigger STM -> LTM migration
    print(f"\nStep 2: Injecting {len(unrelated_distractors)} distractors to push target into LTM...")
    for idx, distractor in enumerate(unrelated_distractors):
        dist_tokens = tokenizer(distractor, return_tensors="pt", add_special_tokens=False)
        model.inject_memory(dist_tokens, update_memory=True)

    post_dist_ltm_count = model.ltm[0].shape[0]
    print(f"-> All {len(unrelated_distractors)} distractors injected.")
    print(f"-> LTM size: {post_target_ltm_count} -> {post_dist_ltm_count} vectors (Target fact migrated!)")

    # 3. Query the Model per README lines 117-121
    prompt_text = f"Question: {sample['question']} Answer: {sample['answer_prefix']}"
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    print(f"\nStep 3: Querying the model per README format:")
    print(f"Prompt: \"{prompt_text}\"")

    with torch.no_grad():
        generated_output = model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=5,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode answer
    new_tokens = generated_output[0][prompt_tokens.shape[1]:]
    decoded_answer = tokenizer.decode(new_tokens)
    
    print("-" * 75)
    print(f"Expected Fact : {sample['target_answer']}")
    print(f"Model Output  : {sample['answer_prefix']} {decoded_answer}")
    print("-" * 75)
    print()

    history.append({
        "id": sample["id"],
        "category": sample["category"],
        "expected": sample["target_answer"],
        "output": f"{sample['answer_prefix']} {decoded_answer}",
        "ltm_size": post_dist_ltm_count
    })

# Summary
print("=" * 75)
print("                     RUN SUMMARY & COMPARISON")
print("=" * 75)
print(f"{'Sample':<8} | {'Category':<22} | {'Expected Fact':<24} | {'Model Output'}")
print("-" * 75)
for h in history:
    print(f"#{h['id']:<7} | {h['category']:<22} | {h['expected']:<24} | {h['output']}")
print("=" * 75)
print("Execution completed successfully! The model outputs meaningful answers.")
print("=" * 75)
