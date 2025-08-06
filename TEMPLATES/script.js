document.addEventListener("DOMContentLoaded", function() {
    const imgExists = document.querySelector("img");
    if (!imgExists) {
        let img = document.createElement("img");
        img.src = "static/images/image.png";
        document.querySelector(".image-container").appendChild(img);
    }
})