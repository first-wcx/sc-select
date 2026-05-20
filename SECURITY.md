# Security Notes

Do not commit credentials, server inventory, SSH connection strings, API keys, model access tokens, or private submission files.

Before publishing a new release, run a repository scan for common secret patterns:

```powershell
rg -n -i "(password|passwd|secret|token|api[_-]?key|ssh|root@|credential|private key)" .
```

Expected benign matches may include tokenizer-related code or documentation explaining that credentials are excluded.
