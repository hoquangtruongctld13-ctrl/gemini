# Troubleshooting Guide

This guide helps you resolve common issues when using WebAI-to-API, particularly cookie-related errors.

---

## Table of Contents

- [Cookie Errors](#cookie-errors)
  - [Error: "Gemini cookies not found or incomplete"](#error-gemini-cookies-not-found-or-incomplete)
  - [Error: "Found __Secure-1PSID (empty value)"](#error-found-__secure-1psid-empty-value)
  - [Error: "Failed to start server. Check if cookies are configured"](#error-failed-to-start-server-check-if-cookies-are-configured)
- [How to Get Gemini Cookies Manually](#how-to-get-gemini-cookies-manually)
- [Windows-Specific Issues](#windows-specific-issues)
- [Proxy Configuration](#proxy-configuration)

---

## Cookie Errors

### Error: "Gemini cookies not found or incomplete"

**Symptoms:**
```
app.utils.browser - INFO - Found __Secure-1PSID (empty value)
app.utils.browser - INFO - Found __Secure-1PSIDTS (empty value)
app.utils.browser - WARNING - Gemini cookies not found or incomplete.
app - ERROR - Gemini cookies not found. Please provide cookies in config.conf or ensure browser is logged in.
```

**Cause:**
This error occurs when the application cannot extract valid Gemini authentication cookies from your browser. Common reasons include:

1. **Browser Not Logged In**: You are not logged into Google Gemini (https://gemini.google.com)
2. **Encrypted Cookies (Windows)**: On Windows, browser cookies are encrypted and decryption may fail
3. **Browser Running**: The browser is currently open, preventing database access
4. **Cookies Expired**: Your Gemini session has expired
5. **Missing Dependencies (Windows)**: Required crypto libraries are not installed

**Solutions:**

#### Solution 1: Manual Cookie Configuration (Recommended)

The most reliable method is to manually extract cookies from your browser and add them to `config.conf`.

See: [How to Get Gemini Cookies Manually](#how-to-get-gemini-cookies-manually)

#### Solution 2: Ensure Browser is Logged In

1. Open your browser (the same one specified in `config.conf`)
2. Go to https://gemini.google.com
3. Log in with your Google account
4. Make sure you can chat with Gemini successfully
5. **Close the browser completely** (important for Windows users)
6. Restart the application

#### Solution 3: Install Windows Crypto Libraries

If you're on Windows and want automatic cookie extraction, install these dependencies:

```bash
pip install pywin32 pycryptodomex
# or with poetry:
poetry add pywin32 pycryptodomex
```

Then **close all browser instances** and restart the application.

---

### Error: "Found __Secure-1PSID (empty value)"

This specific error indicates that the cookie entries exist in your browser database but their values are empty or encrypted and couldn't be decrypted.

**On Windows:**
This usually means the cookie decryption failed. Chrome-based browsers (Chrome, Brave, Edge) encrypt cookies using Windows DPAPI.

**Solution:**
1. **Close ALL browser windows** completely
2. If still failing, use [Manual Cookie Configuration](#how-to-get-gemini-cookies-manually)

---

### Error: "Failed to start server. Check if cookies are configured"

This error appears in the GUI application when the server cannot start due to missing or invalid cookies.

**Solution:**
Follow the [Manual Cookie Configuration](#how-to-get-gemini-cookies-manually) steps below.

---

## How to Get Gemini Cookies Manually

This is the most reliable method and works on all operating systems.

### Step 1: Open Gemini in Your Browser

1. Open your preferred browser
2. Navigate to https://gemini.google.com
3. Log in with your Google account if prompted

### Step 2: Open Developer Tools

- **Chrome/Brave/Edge**: Press `F12` or `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Option+I` (Mac)
- **Firefox**: Press `F12` or `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Option+I` (Mac)
- **Safari**: Press `Cmd+Option+I` (enable Developer menu in Preferences first)

### Step 3: Find the Cookies

1. Go to the **Application** tab (Chrome/Brave/Edge) or **Storage** tab (Firefox)
2. In the left sidebar, expand **Cookies**
3. Click on `https://gemini.google.com`
4. Find these two cookies:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

### Step 4: Copy Cookie Values

1. Click on `__Secure-1PSID`
2. Copy the **Value** field content (it's a long string starting with letters/numbers)
3. Repeat for `__Secure-1PSIDTS`

> **Note:** Cookie values are long strings (usually 50+ characters). Make sure to copy the entire value.

### Step 5: Update config.conf

Edit your `config.conf` file:

```ini
[Cookies]
gemini_cookie_1psid = YOUR_COPIED_1PSID_VALUE_HERE
gemini_cookie_1psidts = YOUR_COPIED_1PSIDTS_VALUE_HERE
```

**Example (values are fake):**
```ini
[Cookies]
gemini_cookie_1psid = g.a000abc123xyz...rest_of_long_string
gemini_cookie_1psidts = sidts-CjEBPVxj...rest_of_long_string
```

### Step 6: Restart the Application

After saving `config.conf`, restart the server:

```bash
poetry run python src/run.py
```

Or restart the GUI application.

---

## Windows-Specific Issues

### Browser Cookie Encryption

On Windows, Chrome-based browsers (Chrome, Brave, Edge) encrypt cookies using:
- Windows Data Protection API (DPAPI)
- AES-256-GCM encryption

**Requirements for automatic extraction:**
1. Close all browser windows
2. Install crypto libraries: `pip install pywin32 pycryptodomex`
3. Run as the same Windows user that owns the browser profile

**If automatic extraction still fails:**
Use the [manual cookie method](#how-to-get-gemini-cookies-manually) instead.

### Firefox on Windows

Firefox doesn't encrypt cookies the same way. If using Firefox:
1. Make sure `config.conf` has `name = firefox` under `[Browser]`
2. Firefox profile must not be in use (close Firefox)

---

## Proxy Configuration

If you're getting network errors (403, connection refused, etc.), you may need to configure a proxy.

Edit `config.conf`:

```ini
[Proxy]
http_proxy = http://127.0.0.1:2334
```

Common scenarios requiring proxy:
- Corporate network restrictions
- Geographic restrictions
- Rate limiting issues

---

## Still Having Issues?

If you've tried all the solutions above and still have problems:

1. **Check Logs**: Run with `--reload` flag for detailed logging
2. **Verify Cookies**: Make sure cookies haven't expired (re-login to Gemini)
3. **Try Different Browser**: Set a different browser in `config.conf`
4. **Open an Issue**: Report the problem at https://github.com/Amm1rr/WebAI-to-API/issues

Include the following in your issue:
- Operating System (Windows/Mac/Linux)
- Browser name and version
- Full error message from logs
- Python version (`python --version`)
