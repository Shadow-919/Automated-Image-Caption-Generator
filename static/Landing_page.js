document.addEventListener("DOMContentLoaded", function () {
  const queryHeaders = document.querySelectorAll(".accordion-header");
  const featureHeaders = document.querySelectorAll(".feature-header");
  const featuresContainer = document.querySelector(".features-container");
  const scrollToTopBtn = document.getElementById("scrollToTop");

  // ================= Common Queries Accordion =================
  queryHeaders.forEach(header => {
    header.addEventListener("click", () => {
      const item = header.parentElement;
      const body = item.querySelector(".accordion-body");
      const isActive = item.classList.contains("active");

      document.querySelectorAll(".accordion-item").forEach(i => {
        i.classList.remove("active");
        const b = i.querySelector(".accordion-body");
        if (b) b.style.maxHeight = null;
      });

      if (!isActive) {
        item.classList.add("active");
        if (body) body.style.maxHeight = body.scrollHeight + "px";
      }
    });
  });

  // ================= Features Accordion =================
  featureHeaders.forEach(header => {
    header.addEventListener("click", () => {
      const item = header.parentElement;
      const body = item.querySelector(".feature-body");
      const isActive = item.classList.contains("active");

      document.querySelectorAll(".feature-item").forEach(i => {
        i.classList.remove("active");
        const b = i.querySelector(".feature-body");
        if (b) b.style.maxHeight = null;
      });

      featuresContainer.classList.remove("expanded");

      if (!isActive) {
        item.classList.add("active");
        if (body) body.style.maxHeight = body.scrollHeight + "px";
        featuresContainer.classList.add("expanded");
      }
    });
  });

  // ================= Scroll to Top Button =================
  window.addEventListener("scroll", () => {
    if (window.scrollY > 0) {
      scrollToTopBtn.style.display = "flex";
    } else {
      scrollToTopBtn.style.display = "none";
    }
  });

  scrollToTopBtn.addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  });
});

document.getElementById("capx-text").addEventListener("click", () => {
  location.reload();
});
