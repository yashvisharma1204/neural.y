/* neural.y — shared blog behaviour: progress bar, TOC, theme toggle */
(function () {
  // ---- theme: restore saved preference ----
  try {
    var saved = localStorage.getItem('neural-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
  } catch (e) {}

  function toggleTheme() {
    var root = document.documentElement;
    var current = root.getAttribute('data-theme');
    if (!current) {
      // fall back to system preference to decide the first flip
      current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    var next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('neural-theme', next); } catch (e) {}
  }

  document.addEventListener('DOMContentLoaded', function () {
    // wire toggle button(s)
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      btn.addEventListener('click', toggleTheme);
    });

    // ---- reading progress ----
    var bar = document.getElementById('progress');
    function onScroll() {
      if (!bar) return;
      var h = document.documentElement;
      var scrolled = h.scrollTop;
      var height = h.scrollHeight - h.clientHeight;
      bar.style.width = (height > 0 ? (scrolled / height) * 100 : 0) + '%';
    }
    if (bar) { window.addEventListener('scroll', onScroll, { passive: true }); onScroll(); }

    // ---- build TOC from section titles ----
    // Prefer explicit ids; otherwise derive from .entry-title inside each section.
    var toc = document.getElementById('toc');
    if (toc) {
      var targets = [];
      var explicit = document.querySelectorAll('section[id], h3[id]');
      if (explicit.length) {
        explicit.forEach(function (el) {
          targets.push({ el: el, text: el.getAttribute('data-toc') || el.textContent.trim() });
        });
      } else {
        document.querySelectorAll('section').forEach(function (sec, i) {
          var t = sec.querySelector('.entry-title');
          if (!t) return;
          if (!sec.id) sec.id = 'sec-' + (i + 1);
          targets.push({ el: sec, text: t.textContent.trim() });
        });
      }
      if (!targets.length) { toc.style.display = 'none'; return; }
      var label = document.createElement('span');
      label.className = 'toc-label';
      label.textContent = 'on this page';
      toc.appendChild(label);
      var links = [];
      targets.forEach(function (tgt) {
        var a = document.createElement('a');
        a.href = '#' + tgt.el.id;
        var clean = tgt.text.replace(/\s+/g, ' ');
        a.textContent = clean.length > 34 ? clean.slice(0, 33) + '…' : clean;
        a.title = clean;
        toc.appendChild(a);
        links.push({ a: a, el: tgt.el });
      });
      // scroll-spy
      function spy() {
        var pos = window.scrollY + 100;
        var current = null;
        links.forEach(function (l) { if (l.el.offsetTop <= pos) current = l; });
        links.forEach(function (l) { l.a.classList.toggle('active', l === current); });
      }
      window.addEventListener('scroll', spy, { passive: true });
      spy();
    }
  });
})();
