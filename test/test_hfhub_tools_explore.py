import os
from pathlib import Path
from shutil import rmtree

from .conftest import run_command, huggingface_hub_access
from pedata.hfhub_tools import explore_datasets


if huggingface_hub_access():

    # ==== global variables and helper functions
    log_filename = "test_exploration_log"
    working_dir = "./huggingface_datasets_info"

    def clean_up(cache: bool = False) -> None:
        """Clean up the working directory for tests"""
        if cache:
            rmtree(Path("./huggingface_datasets_info/.cache/"), ignore_errors=True)
        rmtree(
            Path("./huggingface_datasets_info/test_exploration_log"), ignore_errors=True
        )

    # ==== tests
    def test_explore_datasets():
        clean_up()
        explore_datasets(
            working_dir=working_dir,
            log_filename=log_filename,
            just_testing=True,
        )
        assert os.path.exists(Path(f"{working_dir}/.cache"))
        assert os.path.isfile(Path(f"{working_dir}/{log_filename}.csv"))

        clean_up()

    if False:  # this test deletes the cache so should not be systematically run

        def test_explore_datasets_with_delete_cache():
            clean_up(cache=True)
            explore_datasets(
                working_dir=working_dir,
                log_filename=log_filename,
                delete_cache=True,
                just_testing=True,
            )

            assert not os.path.exists(Path(f"{working_dir}/.cache"))
            assert os.path.isfile(Path(f"{working_dir}/{log_filename}.csv"))

    def test_explore_datasets_command_line():
        run_command(
            "python "
            "src/pedata/hfhub_tools/explore.py "
            "--working_dir huggingface_datasets_info "
            "--log_filename log_test "
            "--just_testing "
        )

    def test_explore_datasets_command_line_2():
        run_command("python src/pedata/hfhub_tools/explore.py --just_testing ")
