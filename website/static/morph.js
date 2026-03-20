(() => {
  const slider = document.getElementById("m");
  const mVal = document.getElementById("mVal");
  const out = document.getElementById("out");
  const status = document.getElementById("status");
  const startDemo = document.getElementById("startDemo");

  let lastReq = 0;
  let inflight = null;
  let debounceTimer = null;
  let demoActive = false;

  function setStatus(text) {
    status.textContent = text || "";
  }

  async function render(m) {
    const reqId = ++lastReq;
    if (inflight) inflight.abort();
    inflight = new AbortController();

    setStatus("Rendering…");
    const t0 = performance.now();

    const res = await fetch(`/api/morph/render?m=${encodeURIComponent(m)}`, { signal: inflight.signal });
    if (!res.ok) throw new Error(await res.text());

    const used = res.headers.get("X-Morph-Used");
    if (used) mVal.textContent = used;

    const blob = await res.blob();
    if (reqId !== lastReq) return;

    const url = URL.createObjectURL(blob);
    out.src = url;
    out.currentTime = 0;
    try { await out.play(); } catch {}

    const dt = Math.round(performance.now() - t0);
    setStatus(`Rendered in ${dt}ms`);
  }

  function enableDemo() {
    demoActive = true;
    slider.disabled = false;
    out.disabled = false;
    startDemo.disabled = true;
    render(Number(slider.value)).catch((e) => {
      setStatus(`Error: ${e.message || e}`);
    });
  }

  startDemo.addEventListener("click", enableDemo);

  slider.addEventListener("input", () => {
    if (!demoActive) return;
    const m = Number(slider.value);
    mVal.textContent = m.toFixed(2);
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      render(m).catch((e) => {
        if (e.name === "AbortError") return;
        setStatus(`Error: ${e.message || e}`);
      });
    }, 150);
  });
})();

