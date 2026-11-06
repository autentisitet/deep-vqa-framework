const API = "/api/v1";

const $ = (id) => document.getElementById(id);

const state = {
  lang: localStorage.getItem("deep-vqa-lang") || "zh",
  mode: "api_key",
  intent: "login",
};

const text = {
  zh: {
    apiKeyDescription: "请输入当前部署的 API Key。认证成功后将跳转到评估工作台。",
    accountDescription: "使用邮箱和密码登录。账号产出的评估记录和可视化会保存在服务端。",
    registerDescription: "创建账号后会自动登录。至少 8 位密码，无需邮箱验证。",
    apiKeyLabel: "API Key",
    emailLabel: "邮箱",
    passwordLabel: "密码",
    language: "English",
    login: "登录并进入工作台",
    register: "创建账号并进入",
    signingIn: "正在登录...",
    registering: "正在创建...",
    switchToRegister: "还没有账号？创建账号",
    switchToLogin: "已有账号？返回登录",
    failed: "操作失败",
    title: "Deep-VQA 工作台",
  },
  en: {
    apiKeyDescription: "Enter the API key for this deployment. You will be redirected to the workspace.",
    accountDescription: "Sign in with your email and password. Your history and artifacts are stored server-side.",
    registerDescription: "Create an account and sign in automatically. At least 8 characters; no email verification.",
    apiKeyLabel: "API key",
    emailLabel: "Email",
    passwordLabel: "Password",
    language: "中文",
    login: "Sign in and open workspace",
    register: "Create account and open",
    signingIn: "Signing in...",
    registering: "Creating...",
    switchToRegister: "No account yet? Create one",
    switchToLogin: "Already registered? Sign in",
    failed: "Request failed",
    title: "Deep-VQA workspace",
  },
};

const messages = () => text[state.lang];
const isAccounts = () => state.mode === "accounts";

function render() {
  const copy = messages();
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  $("loginTitle").textContent = copy.title;
  $("languageButton").textContent = copy.language;
  $("apiKeyFields").hidden = isAccounts();
  $("accountFields").hidden = !isAccounts();
  $("apiKey").required = !isAccounts();
  $("email").required = isAccounts();
  $("password").required = isAccounts();
  $("apiKeyLabel").textContent = copy.apiKeyLabel;
  $("emailLabel").textContent = copy.emailLabel;
  $("passwordLabel").textContent = copy.passwordLabel;

  if (!isAccounts()) {
    $("loginDescription").textContent = copy.apiKeyDescription;
    $("loginButton").textContent = copy.login;
    $("switchModeButton").hidden = true;
    return;
  }

  const registering = state.intent === "register";
  $("loginDescription").textContent = registering ? copy.registerDescription : copy.accountDescription;
  $("loginButton").textContent = registering ? copy.register : copy.login;
  $("switchModeButton").hidden = false;
  $("switchModeButton").textContent = registering ? copy.switchToLogin : copy.switchToRegister;
}

function showError(message) {
  $("loginNotice").textContent = message;
  $("loginNotice").className = "notice show error";
}

function clearError() {
  $("loginNotice").className = "notice";
}

async function readJson(response) {
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : {};
}

function apiError(response, data) {
  const error = new Error(data.error || data.detail || `HTTP ${response.status}`);
  error.code = Number.isInteger(data.code) ? data.code : null;
  error.status = response.status;
  return error;
}

async function authStatus() {
  const response = await fetch(`${API}/auth/status`, { cache: "no-store", credentials: "include" });
  const data = await readJson(response);
  if (!response.ok) throw apiError(response, data);
  return data;
}

function submission() {
  if (isAccounts()) {
    return {
      body: { email: $("email").value.trim(), password: $("password").value },
      path: state.intent === "register" ? "auth/register" : "auth/login",
    };
  }
  return { body: { api_key: $("apiKey").value }, path: "auth/login" };
}

async function submit(event) {
  event.preventDefault();
  const button = $("loginButton");
  const copy = messages();
  const registering = isAccounts() && state.intent === "register";
  clearError();
  button.disabled = true;
  button.textContent = registering ? copy.registering : copy.signingIn;
  try {
    const { body, path } = submission();
    const response = await fetch(`${API}/${path}`, {
      body: JSON.stringify(body),
      credentials: "include",
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const data = await readJson(response);
    if (!response.ok) throw apiError(response, data);
    window.location.replace("/");
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    render();
  }
}

function toggleIntent() {
  state.intent = state.intent === "register" ? "login" : "register";
  clearError();
  render();
}

async function init() {
  render();
  $("loginForm").addEventListener("submit", submit);
  $("switchModeButton").addEventListener("click", toggleIntent);
  $("languageButton").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    localStorage.setItem("deep-vqa-lang", state.lang);
    render();
  });
  try {
    const status = await authStatus();
    state.mode = status.mode || "api_key";
    if (!status.required || status.authenticated) {
      window.location.replace("/");
      return;
    }
    render();
  } catch (error) {
    state.mode = "api_key";
    render();
    showError(error.message);
  }
}

init();
