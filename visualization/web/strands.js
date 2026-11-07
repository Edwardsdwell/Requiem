// Hilos de tinta en canvas. `nivel()` devuelve 0..1 y controla la amplitud (p. ej. volumen del audio).
function strands(cv, nivel, op) {
  op = Object.assign({colores: ["#0E0E0D", "#6B6A67"], n: 6, amp: .5}, op || {});
  const g = cv.getContext("2d"), quieto = matchMedia("(prefers-reduced-motion:reduce)").matches;
  let W, H, t = 0, nv = 0;
  const ajustar = () => {
    const d = devicePixelRatio || 1;
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * d; cv.height = H * d;
    g.setTransform(d, 0, 0, d, 0, 0);
  };
  ajustar(); addEventListener("resize", ajustar);
  (function bucle() {
    t += .016; nv += (nivel() - nv) * .12;
    g.clearRect(0, 0, W, H);
    g.lineWidth = .8;
    op.colores.forEach((c, k) => {
      g.strokeStyle = c;
      for (let s = 0; s < op.n; s++) {
        g.beginPath();
        for (let x = 0; x <= W; x += 8) {
          const u = x / W, env = Math.pow(Math.sin(Math.PI * u), 1.5);
          const y = H / 2 + env * H * op.amp * (.12 + nv) *
            (Math.sin(u * 6 + t * (1 + k * .4) + s * .22 + k) * .6 + Math.sin(u * 11 - t * 1.3 + k * 2 + s * .3) * .4);
          x ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.globalAlpha = .6 - s * .07;
        g.stroke();
      }
    });
    g.globalAlpha = 1;
    if (!quieto) requestAnimationFrame(bucle);
  })();
}
