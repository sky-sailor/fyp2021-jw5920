# Libraries
import pytest
import pathlib
import runpy

# Find the examples folder
examples = pathlib.Path(__file__, '../../../', 'examples').resolve()

# Find all the scripts
scripts_tsne = (examples / 'tsne').glob('**/*.py')


@pytest.mark.parametrize('script', scripts_tsne)
def test_script_execution_widgets(script):
    runpy.run_path(str(script))