"""Exercise the base SDK in an interpreter without optional model or execution packages."""

import subprocess
import sys
import textwrap

import pytest


def run_without_packages(packages: list[str], source: str) -> None:
    """Run a consumer with selected optional packages unavailable."""
    isolation = f"""
import importlib.abc
import sys

class MissingOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {packages!r}:
            raise ModuleNotFoundError(f'No module named {{fullname!r}}', name=fullname)

sys.meta_path.insert(0, MissingOptionalPackages())
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", isolation + textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "packages",
    [
        ["instructor"],
        ["langchain", "langgraph", "ormsgpack"],
        ["instructor", "langchain", "langgraph", "ormsgpack"],
    ],
)
def test_custom_models_and_tools_without_optional_packages(packages: list[str]) -> None:
    """Custom LangChain clients and tools remain usable without the optional engines."""
    run_without_packages(
        packages,
        """
        from unittest.mock import MagicMock
        from pydantic import BaseModel
        from portia import Config, Message, Tool, ToolRunContext
        from portia.model import LangChainGenerativeModel
        from portia.plan import Plan, PlanContext, Step, Variable

        class Answer(BaseModel):
            value: str

        client = MagicMock()
        client.with_structured_output.return_value.invoke.return_value = {'value': 'ok'}
        model = LangChainGenerativeModel(client=client, model_name='custom')
        config = Config.from_default(default_model=model)
        response = config.get_default_model().get_structured_response(
            [Message(role='user', content='hello')], Answer
        )
        assert response == Answer(value='ok')

        class EchoTool(Tool):
            id: str = 'echo'
            name: str = 'Echo'
            description: str = 'Echo a value'
            output_schema: tuple[str, str] = ('str', 'Echoed value')
            def run(self, ctx: ToolRunContext, value: str) -> str:
                return value

        assert EchoTool().run(None, 'hello') == 'hello'
        """,
    )


def test_native_model_explains_missing_instructor_extra() -> None:
    """Selecting an Instructor-backed adapter reports how to enable it."""
    run_without_packages(
        ["instructor"],
        """
        from pydantic import SecretStr
        from portia.model import OpenAIGenerativeModel
        try:
            OpenAIGenerativeModel(model_name='gpt-4o', api_key=SecretStr('test'))
        except ImportError as error:
            assert 'portia-sdk-python[instructor]' in str(error), str(error)
        else:
            raise AssertionError('Expected an actionable missing-extra error')
        """,
    )


def test_execution_explains_missing_execution_extra() -> None:
    """Selecting the execution engine reports how to enable it."""
    run_without_packages(
        ["langchain", "langgraph", "ormsgpack"],
        """
        import portia
        try:
            from portia import Portia
        except ImportError as error:
            assert 'portia-sdk-python[execution]' in str(error), str(error)
        else:
            raise AssertionError('Expected an actionable missing-extra error')
        """,
    )
