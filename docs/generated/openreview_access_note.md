# OpenReview Access Note

For future OpenReview research in this workspace:

- Prefer the in-app/local browser that the user has already verified.
- Do not start by repeatedly trying `curl`, OpenReview API, or external browserless access, because OpenReview often triggers browser verification / Cloudflare challenge.
- If a new OpenReview page is blocked, ask the user to verify it in the local browser once, then continue reading via the browser page state.
- Use API/search only as a secondary supplement when browser access is unavailable or the user explicitly asks for it.

中文备注：

以后在这个 workspace 里调研 OpenReview，默认走本地浏览器 / in-app browser。不要先反复尝试 API 或 curl，避免来回卡验证。若页面再次触发验证墙，直接请用户在浏览器里验证一次，再继续读取页面内容。
