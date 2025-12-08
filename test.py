"""
Test script to verify models are working correctly
Run this before starting the server to check everything
"""
import sys
import os

print("=" * 60)
print("FedSearch-NLP Model Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from models import RetrieverModel, GeneratorModel
    print("   ✓ Models imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test Retriever
print("\n2. Testing Retriever Model...")
try:
    retriever = RetrieverModel()
    test_texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = retriever.encode(test_texts)
    print(f"   ✓ Retriever working. Embedding dim: {embeddings.shape}")
except Exception as e:
    print(f"   ✗ Retriever error: {e}")
    import traceback
    traceback.print_exc()

# Test Generator
print("\n3. Testing Generator Model with LoRA...")
try:
    generator = GeneratorModel(use_lora=True)
    
    # Check trainable parameters
    trainable_count = sum(p.numel() for p in generator.model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in generator.model.parameters())
    
    print(f"   Total parameters: {total_count:,}")
    print(f"   Trainable parameters: {trainable_count:,}")
    print(f"   Trainable %: {100 * trainable_count / total_count:.2f}%")
    
    if trainable_count == 0:
        print("   ⚠ WARNING: No trainable parameters!")
        print("   Trying alternative LoRA setup...")
        
        # Try with different target modules
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForSeq2SeqLM
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        
        # Print available modules
        print("\n   Available modules in model:")
        for name, module in base_model.named_modules():
            if 'attention' in name.lower() or 'dense' in name.lower():
                print(f"     - {name}: {type(module).__name__}")
        
        # Try with all linear layers in attention
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "k", "v", "o"],  # All attention projections
            inference_mode=False,
            bias="none"
        )
        
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
        
        trainable_count_new = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"   New trainable count: {trainable_count_new:,}")
        
        if trainable_count_new > 0:
            print("   ✓ LoRA working with extended target modules!")
            print("   Update models.py to use: target_modules=['q', 'k', 'v', 'o']")
    else:
        print("   ✓ Generator with LoRA working correctly")
    
except Exception as e:
    print(f"   ✗ Generator error: {e}")
    import traceback
    traceback.print_exc()

# Test generation
print("\n4. Testing text generation...")
try:
    test_prompt = "Question: What is machine learning? Answer:"
    answer = generator.generate(test_prompt, max_length=50)
    print(f"   ✓ Generated: {answer[:100]}...")
except Exception as e:
    print(f"   ✗ Generation error: {e}")
    import traceback
    traceback.print_exc()

# Test state dict extraction
print("\n5. Testing state dict extraction...")
try:
    state_dict = generator.get_trainable_state_dict()
    print(f"   ✓ State dict extracted. Keys: {len(state_dict)}")
    if state_dict:
        first_key = list(state_dict.keys())[0]
        print(f"   Sample key: {first_key}")
except Exception as e:
    print(f"   ✗ State dict error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nIf all tests passed, you can start the server with:")
print("  python server.py")
print("\nIf LoRA has 0 trainable parameters, the code will fall back")
print("to training the entire model (slower but will work).")
print("=" * 60)