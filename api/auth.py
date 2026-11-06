"""Server-rendered sign-in and account-creation forms (no client-side JavaScript)."""
from __future__ import annotations

from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.auth import AuthError

router = APIRouter()


def _page(mode: str, error: str = "", message: str = "", allow_signup: bool = False) -> HTMLResponse:
    signup = mode == "signup" and allow_signup
    title = "Create your account" if signup else "Welcome back"
    action = "/signup" if signup else "/login"
    switch = ("<p>Already have an account? <a href='/login'>Sign in</a></p>" if signup else
              ("<p>New here? <a href='/signup'>Create an account</a></p>" if allow_signup else ""))
    notice = f"<div class='notice error'>{escape(error)}</div>" if error else (f"<div class='notice ok'>{escape(message)}</div>" if message else "")
    return HTMLResponse(f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title} · DataPilot</title><link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
<link href='https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Public+Sans:wght@400;500;600&display=swap' rel='stylesheet'>
<style>:root{{--ink:#171a25;--plum:#6658d4;--paper:#f6f5f2;--line:#e4e2dd}}*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 85% 5%,#e8e5ff,transparent 28%),var(--paper);font:15px/1.5 'Public Sans',sans-serif;color:var(--ink)}}main{{width:min(100% - 32px,420px)}}.brand{{display:flex;align-items:center;gap:10px;justify-content:center;font:700 25px 'Bricolage Grotesque';letter-spacing:-.04em;margin-bottom:22px}}.mark{{display:grid;place-items:center;width:32px;height:32px;border-radius:10px;background:var(--plum);color:#fff}}.card{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:31px;box-shadow:0 18px 50px #29244712}}h1{{font:700 30px/1.05 'Bricolage Grotesque';letter-spacing:-.04em;margin:0 0 7px}}.sub{{margin:0 0 23px;color:#70717a}}label{{display:block;font-weight:600;font-size:13px;margin:14px 0 6px}}input{{width:100%;border:1px solid #d9d6d2;border-radius:9px;padding:11px 12px;font:inherit}}input:focus{{outline:2px solid #c5befb;border-color:var(--plum)}}button{{width:100%;margin-top:21px;border:0;border-radius:9px;background:var(--plum);color:#fff;padding:12px;font:600 14px 'Public Sans';cursor:pointer}}button:hover{{background:#5044b5}}.notice{{padding:9px 11px;border-radius:8px;font-size:13px;margin:0 0 14px}}.error{{background:#fff0f1;color:#a33243}}.ok{{background:#edfbf1;color:#237344}}.card p{{text-align:center;color:#73737c;font-size:13px;margin:20px 0 0}}a{{color:#5548bf;font-weight:600;text-decoration:none}}</style></head><body><main><div class='brand'><span class='mark'>✦</span>DataPilot</div><section class='card'><h1>{title}</h1><p class='sub'>Sign in to your private data-analysis workspace.</p>{notice}<form method='post' action='{action}'><label for='username'>Username</label><input id='username' name='username' autocomplete='username' required><label for='password'>Password</label><input id='password' name='password' type='password' autocomplete='{ 'new-password' if signup else 'current-password' }' minlength='8' required><button type='submit'>{'Create account' if signup else 'Sign in'}</button></form>{switch}</section></main></body></html>""")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _page("login", message="Account created. You can sign in now." if request.query_params.get("created") else "",
                 allow_signup=request.app.state.settings.allow_signup)


@router.post("/login")
async def login(request: Request):
    form = await request.form()
    username = request.app.state.auth.verify(str(form.get("username", "")), str(form.get("password", "")))
    if not username:
        return _page("login", error="Incorrect username or password.", allow_signup=request.app.state.settings.allow_signup)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("dp_auth", request.app.state.auth.issue_token(username), httponly=True, samesite="lax",
                        secure=request.app.state.settings.auth_cookie_secure, max_age=43200)
    return response


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    if not request.app.state.settings.allow_signup:
        return _page("login", error="Account creation is disabled for this deployment.")
    return _page("signup", allow_signup=True)


@router.post("/signup")
async def signup(request: Request):
    if not request.app.state.settings.allow_signup:
        return _page("login", error="Account creation is disabled for this deployment.")
    form = await request.form()
    try:
        request.app.state.auth.create_user(str(form.get("username", "")), str(form.get("password", "")))
    except AuthError as exc:
        return _page("signup", error=str(exc), allow_signup=True)
    return RedirectResponse("/login?created=1", status_code=303)


@router.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("dp_auth")
    return response
