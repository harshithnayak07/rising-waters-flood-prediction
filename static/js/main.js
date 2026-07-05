document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector(".prediction-form");

    if (!form) {
        return;
    }

    form.addEventListener("submit", function (event) {
        const inputs = form.querySelectorAll("input");

        for (const input of inputs) {
            if (input.value.trim() === "") {
                event.preventDefault();
                alert("Please fill all input fields before submitting the prediction form.");
                input.focus();
                return;
            }
        }
    });
});
