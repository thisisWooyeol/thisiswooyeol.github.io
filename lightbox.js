// Get the lightbox
var lightbox = document.getElementById("lightbox");

var lightboxImg = document.getElementById("lightbox-content");

// Open dynamically rendered zoomable images in the lightbox
document.addEventListener("click", function (event) {
  var image = event.target.closest(".zoomable-image");

  if (image) {
    lightbox.style.display = "block";
    lightboxImg.src = image.src;
  }
});

// Get the <span> element that closes the lightbox
var span = document.getElementById("lightbox-close");

// When the user clicks on <span> (x), close the lightbox
span.onclick = function () {
  lightbox.style.display = "none";
};

// When the user clicks anywhere outside of the lightbox, close it
window.onclick = function (event) {
  if (event.target == lightbox) {
    lightbox.style.display = "none";
  }
};
