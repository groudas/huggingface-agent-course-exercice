# smoke_test.py
# Lightweight smoke test that runs the orchestrator flow locally with a mocked
# LLM function to avoid external network calls. Verifies that ValidatorNode will
# accept a good YAML final answer and that the flow transitions reach FinishNode.

from app import create_orchestrator_flow, SynthesizerNode, FinalAnswerNode, ValidatorNode, FinishNode

# Monkeypatch the LLM call used in app.py
import app

responses = {
    # Simulate orchestrator responding with action: final_answer
    'orchestrator': "```yaml\naction: final_answer\nreason: confident\n```",
    # Synthesizer returns a fenced YAML final answer
    'synthesizer': "```yaml\nfinal_answer: 42\n```",
    # FinalAnswerNode may echo back same YAML
    'final': "```yaml\nfinal_answer: 42\n```",
    # Repair response expected by ValidatorNode
    'repair': "```yaml\nfinal_answer: 42\n```",
}

# Simple stub that picks response by inspecting system prompt role or content
def mock_call_perplexity_api(messages, max_tokens=300, model=None):
    # messages is a list of dicts; pick by analyzing the system prompt
    sys = messages[0]['content'] if messages and 'content' in messages[0] else ''
    usr = messages[-1]['content'] if messages else ''
    if 'orchestrator' in sys.lower() or 'orchestrator' in usr.lower():
        return responses['orchestrator']
    if 'concise synthesizer' in sys.lower() or 'synthesizer' in sys.lower():
        return responses['synthesizer']
    if 'final-answer agent' in sys.lower() or 'final-answer' in sys.lower() or 'final answer' in sys.lower():
        return responses['final']
    if 'repair agent' in sys.lower() or 'please return the repaired yaml' in usr.lower():
        return responses['repair']
    # Default fallback
    return responses['synthesizer']

# Patch
app.call_perplexity_api = mock_call_perplexity_api


def run_smoke():
    flow = create_orchestrator_flow()
    shared = {'question': 'How many test items?', 'current_question': 'How many test items?', 'reasoning': ''}
    flow.run(shared)
    print('Shared after flow run:', shared)
    assert 'final_answer' in shared, 'No final_answer produced'
    assert shared.get('final_answer') == 42 or shared.get('final_answer') == '42', 'Final answer not parsed as 42'
    assert 'final_answer_valid' in shared and shared['final_answer_valid'] is True, 'Final answer was not validated'
    print('Smoke test passed.')

if __name__ == '__main__':
    run_smoke()
