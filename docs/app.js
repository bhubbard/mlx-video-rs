document.addEventListener("DOMContentLoaded", () => {
  // --- 1. Theme Toggle with Zero-FOUC support & OS change listener ---
  const themeToggleBtn = document.getElementById("theme-toggle");
  const metaColorScheme = document.querySelector('meta[name="color-scheme"]');

  function getPreferredTheme() {
    const stored = localStorage.getItem("theme");
    if (stored) return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (metaColorScheme) {
      metaColorScheme.content = theme === "dark" ? "dark" : "light";
    }
    const icon = themeToggleBtn.querySelector(".theme-icon");
    if (icon) {
      icon.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  const currentTheme = getPreferredTheme();
  applyTheme(currentTheme);

  themeToggleBtn.addEventListener("click", () => {
    const active = document.documentElement.getAttribute("data-theme") || "dark";
    const next = active === "dark" ? "light" : "dark";
    localStorage.setItem("theme", next);
    applyTheme(next);
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem("theme")) {
      applyTheme(e.matches ? "dark" : "light");
    }
  });

  // --- 2. Copy Code to Clipboard ---
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetId = btn.getAttribute("data-target");
      const targetEl = document.getElementById(targetId);
      if (!targetEl) return;

      const text = targetEl.innerText || targetEl.textContent;
      try {
        await navigator.clipboard.writeText(text.trim());
        const originalText = btn.innerHTML;
        btn.innerHTML = `<span>✓ Copied!</span>`;
        btn.style.borderColor = "var(--accent-cyan)";
        btn.style.color = "var(--accent-cyan)";
        setTimeout(() => {
          btn.innerHTML = originalText;
          btn.style.borderColor = "";
          btn.style.color = "";
        }, 2000);
      } catch (err) {
        console.error("Clipboard copy failed: ", err);
      }
    });
  });

  // --- 3. Interactive CLI Playground Generator ---
  const modelSelect = document.getElementById("pg-model");
  const promptInput = document.getElementById("pg-prompt");
  const framesRange = document.getElementById("pg-frames");
  const framesVal = document.getElementById("pg-frames-val");
  const stepsRange = document.getElementById("pg-steps");
  const stepsVal = document.getElementById("pg-steps-val");
  const cfgRange = document.getElementById("pg-cfg");
  const cfgVal = document.getElementById("pg-cfg-val");
  const resSelect = document.getElementById("pg-res");
  const imageInputGroup = document.getElementById("pg-image-group");
  const imageInput = document.getElementById("pg-image");
  const cliOutput = document.getElementById("pg-cli-output");

  function updatePlayground() {
    const model = modelSelect.value;
    const prompt = promptInput.value.replace(/"/g, '\\"');
    const frames = framesRange.value;
    const steps = stepsRange.value;
    const cfg = cfgRange.value;
    const [w, h] = resSelect.value.split("x");
    const image = imageInput ? imageInput.value.trim() : "";

    framesVal.textContent = frames;
    stepsVal.textContent = steps;
    cfgVal.textContent = parseFloat(cfg).toFixed(1);

    if (model.includes("i2v")) {
      imageInputGroup.style.display = "block";
    } else {
      imageInputGroup.style.display = "none";
    }

    let cmd = "";
    if (model.startsWith("ltx")) {
      cmd = `mlx-video ltx generate \\\n  --prompt "${prompt}" \\\n  --pipeline ${model === "ltx2_distilled" ? "distilled" : "dev"} \\\n  --width ${w} \\\n  --height ${h} \\\n  --num-frames ${frames} \\\n  --steps ${steps} \\\n  --output output.mp4`;
    } else {
      let imageFlag = (model.includes("i2v") && image) ? `  --image ${image} \\\n` : "";
      cmd = `mlx-video wan generate \\\n  --prompt "${prompt}" \\\n${imageFlag}  --model ${model} \\\n  --width ${w} \\\n  --height ${h} \\\n  --num-frames ${frames} \\\n  --steps ${steps} \\\n  --guide-scale ${cfg} \\\n  --output output.mp4`;
    }

    cliOutput.textContent = cmd;
  }

  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      if (modelSelect.value === "ltx2_distilled") {
        stepsRange.value = 8;
        cfgRange.value = 1.0;
      } else if (modelSelect.value.startsWith("wan")) {
        stepsRange.value = 40;
        cfgRange.value = 6.0;
      }
      updatePlayground();
    });

    promptInput.addEventListener("input", updatePlayground);
    framesRange.addEventListener("input", updatePlayground);
    stepsRange.addEventListener("input", updatePlayground);
    cfgRange.addEventListener("input", updatePlayground);
    resSelect.addEventListener("change", updatePlayground);
    if (imageInput) imageInput.addEventListener("input", updatePlayground);

    updatePlayground();
  }

  // --- 4. Models Matrix Filter Tabs ---
  const filterTabs = document.querySelectorAll(".filter-tab");
  const modelRows = document.querySelectorAll("tbody tr");

  filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      filterTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const filter = tab.getAttribute("data-filter");
      modelRows.forEach((row) => {
        const category = row.getAttribute("data-category");
        if (filter === "all" || category === filter) {
          row.style.display = "";
        } else {
          row.style.display = "none";
        }
      });
    });
  });
});
