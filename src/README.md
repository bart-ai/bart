# bart-ai: web

The bart-ai web is powered by https://streamlit.io/ and deployed onto https://fly.io/.

## Run locally

```sh
# on a venv
$ pip install -r requirements.txt
$ streamlit run web.py
```

## Deployment

On each push to `master` the CI automatically deploys the new version to https://bart.fly.dev/, by running the project [Dockerfile](../Dockerfile) and reading the specs from the project [fly.toml](../fly.toml).

To manually deploy, install the [fly command-line tool](https://fly.io/docs/hands-on/install-flyctl/) and run `flyctl deploy` after authentication.
