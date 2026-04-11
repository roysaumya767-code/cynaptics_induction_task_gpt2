def ask_model(question):
    prompt = f"""Below is an instruction that desc ribes a task.

### Instruction:
{question}

### Response:
"""                 
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id

    )
    text = tokenizer.decode(output[0], skip_special_tokens = True)
    print(text)


    
