// Get the lightbox
var lightbox = document.getElementById("lightbox");

// Get the image and insert it inside the lightbox
var lightboxImg = document.getElementById("lightbox-content");
var images = document.getElementsByClassName("zoomable-image");

for (var i = 0; i < images.length; i++) {
  images[i].onclick = function () {
    lightbox.style.display = "block";
    lightboxImg.src = this.src;
  };
}

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
