from app import call_perplexity_api

print('Testing Perplexity helper...')
try:
    out = call_perplexity_api([
        {"role":"system","content":"You are a test assistant."},
        {"role":"user","content":"Say hello."}
    ], max_tokens=20)
    print('Perplexity responded:', out)
except Exception as e:
    print('Perplexity call failed:', e)
