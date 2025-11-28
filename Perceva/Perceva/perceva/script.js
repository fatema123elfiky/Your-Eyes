// Modal
const authModal = document.getElementById("authModal");
const getStartedBtns = document.querySelectorAll(".getStartedBtn");
const closeModalBtn = document.getElementById("closeModal");
const loginLink = document.getElementById("loginLink");
const formTitle = document.getElementById("formTitle");

getStartedBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    authModal.classList.remove("hidden");
    formTitle.textContent = "Sign Up";
  });
});

closeModalBtn.addEventListener("click", () => {
  authModal.classList.add("hidden");
});

loginLink.addEventListener("click", (e) => {
  e.preventDefault();
  formTitle.textContent = "Log In";
});

