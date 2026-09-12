import pytest
from lip import train_stream


@pytest.mark.parametrize('steps',['0','-1'])
def test_preflight_cannot_expand_nonpositive_steps_to_long_training(monkeypatch,steps):
    monkeypatch.setattr('sys.argv',['train_stream','--config','does-not-exist.yaml','--init-from','missing.pt',
        '--output','must-not-be-created','--data-root','not-read','--preflight','--max-steps',steps])
    with pytest.raises(SystemExit) as exc:train_stream.main()
    assert exc.value.code==2
