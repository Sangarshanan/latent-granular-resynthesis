(() => {
  const slider = document.getElementById("k");
  const kVal = document.getElementById("kVal");
  const out = document.getElementById("out");
  const status = document.getElementById("status");

  let lastReq = 0;
  let inflight = null;
  let debounceTimer = null;

  function setStatus(text) {
    status.textContent = text || "";
  }

  async function render(k) {
    const reqId = ++lastReq;
    if (inflight) inflight.abort();
    inflight = new AbortController();

    setStatus("Rendering…");
    const t0 = performance.now();

    const res = await fetch(`/api/demo/render?k=${k}`, { signal: inflight.signal });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();

    if (reqId !== lastReq) return;
    const url = URL.createObjectURL(blob);
    out.src = url;
    out.currentTime = 0;
    try { await out.play(); } catch {}

    const dt = Math.round(performance.now() - t0);
    setStatus(`Rendered in ${dt}ms`);
  }


  const startDemo = document.getElementById("startDemo");
  slider.disabled = true;
  startDemo.addEventListener("click", () => {
    slider.disabled = false;
    startDemo.disabled = true;
    render(Number(slider.value)).catch((e) => {
      setStatus(`Error: ${e.message || e}`);
    });
  });

  slider.addEventListener("input", () => {
    if (slider.disabled) return;
    const k = Number(slider.value);
    kVal.textContent = String(k);
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      render(k).catch((e) => {
        if (e.name === "AbortError") return;
        setStatus(`Error: ${e.message || e}`);
      });
    }, 150);
  });

  // Optional: pool morph demo (if present on the page)
  const morphSlider = document.getElementById("m");
  const mVal = document.getElementById("mVal");
  const outMorph = document.getElementById("outMorph");
  const statusMorph = document.getElementById("statusMorph");

  if (morphSlider && mVal && outMorph && statusMorph) {
    const startMorphDemo = document.getElementById("startMorphDemo");
    morphSlider.disabled = true;
    let lastReqMorph = 0;
    let inflightMorph = null;
    let debounceMorph = null;
    let morphDemoActive = false;

    function setStatusMorph(text) {
      statusMorph.textContent = text || "";
    }

    async function renderMorph(m) {
      const reqId = ++lastReqMorph;
      if (inflightMorph) inflightMorph.abort();
      inflightMorph = new AbortController();

      setStatusMorph("Rendering…");
      const t0 = performance.now();

      const res = await fetch(`/api/morph/render?m=${encodeURIComponent(m)}`, { signal: inflightMorph.signal });
      if (!res.ok) throw new Error(await res.text());

      const used = res.headers.get("X-Morph-Used");
      if (used) mVal.textContent = used;

      const blob = await res.blob();
      if (reqId !== lastReqMorph) return;

      const url = URL.createObjectURL(blob);
      outMorph.src = url;
      outMorph.currentTime = 0;
      try { await outMorph.play(); } catch {}

      const dt = Math.round(performance.now() - t0);
      setStatusMorph(`Rendered in ${dt}ms`);
    }

    function enableMorphDemo() {
      morphDemoActive = true;
      morphSlider.disabled = false;
      startMorphDemo.disabled = true;
      renderMorph(Number(morphSlider.value)).catch((e) => {
        setStatusMorph(`Error: ${e.message || e}`);
      });
    }

    startMorphDemo.addEventListener("click", enableMorphDemo);

    morphSlider.addEventListener("input", () => {
      if (!morphDemoActive) return;
      const m = Number(morphSlider.value);
      mVal.textContent = m.toFixed(2);
      if (debounceMorph) clearTimeout(debounceMorph);
      debounceMorph = setTimeout(() => {
        renderMorph(m).catch((e) => {
          if (e.name === "AbortError") return;
          setStatusMorph(`Error: ${e.message || e}`);
        });
      }, 150);
    });
  }
})();
