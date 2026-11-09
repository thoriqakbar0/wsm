# Archived verification workflow

`verify.yml.disabled` is a saved template, outside GitHub Actions discovery. It has push and manual triggers, with no pull-request trigger. Repository-level Actions also remains disabled.

Restoring automation requires explicitly enabling Actions and copying the template to `.github/workflows/verify.yml`.
