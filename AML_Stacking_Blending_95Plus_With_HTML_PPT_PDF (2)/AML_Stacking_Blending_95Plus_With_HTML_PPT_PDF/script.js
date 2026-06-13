function qs(sel, root=document){ return root.querySelector(sel); }
function qsa(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

function copyCode(id){
  const el = document.getElementById(id);
  if(!el) return;
  navigator.clipboard.writeText(el.innerText).then(() => toast("Code copié."));
}

function toast(msg){
  const t = document.createElement("div");
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;right:22px;bottom:22px;z-index:9999;
    background:#2563EB;color:white;padding:12px 16px;border-radius:14px;
    font-weight:800;box-shadow:0 16px 40px rgba(37,99,235,.25)
  `;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 1700);
}

function toggleTheme(){
  document.documentElement.classList.toggle("dark");
  localStorage.setItem("aml_theme", document.documentElement.classList.contains("dark") ? "dark" : "light");
}

function restoreTheme(){
  if(localStorage.getItem("aml_theme") === "dark"){
    document.documentElement.classList.add("dark");
  }
}

function updateProgress(){
  const h = document.documentElement;
  const total = h.scrollHeight - h.clientHeight;
  const value = total <= 0 ? 0 : (h.scrollTop / total) * 100;
  const bar = qs(".progress");
  if(bar) bar.style.width = value + "%";
}

function setupTabs(){
  qsa(".tabs").forEach(tabs => {
    const buttons = qsa(".tab-buttons button", tabs);
    const panels = qsa(".tab-panel", tabs);
    buttons.forEach((b, i) => b.addEventListener("click", () => {
      buttons.forEach(x => x.classList.remove("active"));
      panels.forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      if(panels[i]) panels[i].classList.add("active");
    }));
  });
}

function setupMobile(){
  const side = qs(".sidebar");
  qsa(".mobile-menu").forEach(btn => {
    btn.addEventListener("click", () => {
      if(side) side.classList.toggle("open");
    });
  });
  qsa(".nav a").forEach(a => {
    a.addEventListener("click", () => {
      if(window.innerWidth < 980 && side) side.classList.remove("open");
    });
  });
}

function setupActiveNav(){
  const current = location.pathname.split("/").pop() || "index.html";
  qsa(".nav a").forEach(a => {
    const href = a.getAttribute("href") || "";
    if(href.endsWith(current)) a.classList.add("active");
  });
}

function setupBackToTop(){
  const b = document.createElement("button");
  b.className = "btn primary";
  b.textContent = "↑";
  b.setAttribute("aria-label", "Retour en haut");
  b.style.cssText = "position:fixed;right:22px;bottom:22px;z-index:800;display:none;width:44px;height:44px;padding:0;border-radius:16px";
  document.body.appendChild(b);
  b.onclick = () => window.scrollTo({top:0, behavior:"smooth"});
  window.addEventListener("scroll", () => {
    b.style.display = window.scrollY > 700 ? "inline-flex" : "none";
  });
}

function openVS(path){
  navigator.clipboard.writeText(path).then(() => toast("Chemin copié pour VS Code : " + path));
}

document.addEventListener("DOMContentLoaded", () => {
  restoreTheme();
  setupTabs();
  setupMobile();
  setupActiveNav();
  setupBackToTop();
  updateProgress();
});
document.addEventListener("scroll", updateProgress);
