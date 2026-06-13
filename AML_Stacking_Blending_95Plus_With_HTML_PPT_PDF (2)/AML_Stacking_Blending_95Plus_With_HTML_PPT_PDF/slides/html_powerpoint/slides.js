
let index = 0;
const slides = [...document.querySelectorAll('.slide')];
function show(i){ index = Math.max(0, Math.min(slides.length-1, i)); slides[index].scrollIntoView({behavior:'smooth', block:'center'}); }
function nextSlide(){ show(index+1); }
function prevSlide(){ show(index-1); }
document.addEventListener('keydown', e => { if(e.key==='ArrowRight') nextSlide(); if(e.key==='ArrowLeft') prevSlide(); });
const obs = new IntersectionObserver(entries => { entries.forEach(e => { if(e.isIntersecting){ index = slides.indexOf(e.target); }}); }, {threshold:.55});
slides.forEach(s => obs.observe(s));
