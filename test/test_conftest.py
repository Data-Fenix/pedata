from .conftest import huggingface_hub_access, run_command


def test_huggingface_hub_access():
    response = huggingface_hub_access()
    assert response is True


def test_run_command():
    response = run_command("echo Hello Exazyme!")
    assert response.stdout == "Hello Exazyme!\n"
    assert response.stderr == ""
