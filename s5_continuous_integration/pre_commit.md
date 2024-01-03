![Logo](../figures/icons/precommit.png){ align=right width="130"}

# Pre-commit

---

One of the cornerstones of working with git is remembering to commit your work often. Often committing makes sure
that it is easier to identify and revert unwanted changes that you have introduced, because the code changes becomes
smaller per commit.

However, as you hopefully already seen in the course there are a lot of mental task to do before you actually write
`git commit` in the terminal. The most basic thing is of course making sure that you have saved all your changes, and
you are not committing a not up-to-date file. However, this also includes tasks such as styling, formatting, making
sure all tests succeeds etc. All these mental to-do notes does not mix well with the principal of remembering to commit
often, because you in principal have to do them every time.

The obvious solution to this problem is to automate all or some of our mental task every time that we do a commit. This
is where *pre-commit hooks* comes into play, as they can help us attach additional tasks that should be run every time
that we do a `git commit`.

## Configuration

Pre-commit simply works by inserting whatever workflow we want to automate in between whenever we do a `git commit` and
afterwards would do a `git push`.

<figure markdown>
![Image](../figures/pre_commit.png){ width="700" }
<figcaption>
<a href="https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/"> Image credit </a>
</figcaption>
</figure>

The system works by looking for a file called `.pre-commit-config.yaml` that we can configure. If we execute

```bash
pre-commit sample-config | out-file .pre-commit-config.yaml -encoding utf8
```

you should get a sample file that looks like

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```

the file structure is very simple:

* It starts by listing the repositories where we want to get our pre-commits from, in this case
  <https://github.com/pre-commit/pre-commit-hooks>. This repository contains a large collection of pre-commit hooks.
* Next we need to defined what pre-commit hooks that we want to get by specifying the `id` of the different hooks.
  The `id` corresponds to an `id` in this file:
  <https://github.com/pre-commit/pre-commit-hooks/blob/master/.pre-commit-hooks.yaml>

When we are done defining our `.pre-commit-config.yaml` we just need to install it

```bash
pre-commit install
```

this will make sure that the file is automatically executed whenever we run `git commit`

### ❔ Exercises

1. Install pre-commit

    ```bash
    pip install pre-commit
    ```

2. Next create the sample file

    ```bash
    pre-commit sample-config > .pre-commit-config.yaml
    ```

3. The sample file already contains 4 hooks. Make sure you understand what each do and if you need them at all.

4. `pre-commit` works by hooking into the `git commit` command, running whenever that command is run. For this to work,
    we need to install the hooks into `git commit`. Run

    ```bash
    pre-commit install
    ```

    to do this.

5. Try to commit your recently created `.pre-commit-config.yaml` file. You will likely not do anything, because
    `pre-commit` only check files that are being committed. Instead try to run

    ```bash
    pre-commit run --all-files
    ```

    that will check every file in your repository.

6. Try adding at least another check from the [base repository](https://github.com/pre-commit/pre-commit-hooks) to your
    `.pre-commit-config.yaml` file.

7. If you have completed the optional module
    [M7 on good coding practice](../s2_organisation_and_version_control/good_coding_practice.md) you will have learned
    about the linter `ruff`. `ruff` comes with its own [pre-commit hook](https://github.com/astral-sh/ruff-pre-commit).
    Try adding that to your `.pre-commit-config.yaml` file and see what happens when you try to commit files.

8. (Optional) Add more hooks to your `.pre-commit-config.yaml`.

9. Sometimes you are in a hurry, so make sure that you also can do commits without running `pre-commit` e.g.

    ```bash
    git commit -m <message> --no-verify
    ```

10. Finally, figure out how to disable `pre-commit` again (if you get tired of it).

That was all about how `pre-commit` can be used to automate tasks. If you want to deep dive more into the topic you
can checkout this [page](https://pre-commit.com/#python) on how to define your own `pre-commit` hooks.
