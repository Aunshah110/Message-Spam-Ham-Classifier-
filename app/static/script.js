const messageInput = document.getElementById("message");
const checkButton = document.getElementById("checkButton");

const characterCount = document.getElementById(
    "characterCount"
);

const loading = document.getElementById(
    "loading"
);

const resultCard = document.getElementById(
    "resultCard"
);

const prediction = document.getElementById(
    "prediction"
);

const resultIcon = document.getElementById(
    "resultIcon"
);

const resultMessage = document.getElementById(
    "resultMessage"
);

const score = document.getElementById(
    "score"
);

const scoreProgress = document.getElementById(
    "scoreProgress"
);


/* --------------------------------------------------
   Character counter
-------------------------------------------------- */

messageInput.addEventListener(
    "input",
    () => {

        const length =
            messageInput.value.length;

        characterCount.textContent =
            `${length} / 500`;

    }
);


/* --------------------------------------------------
   Check message
-------------------------------------------------- */

checkButton.addEventListener(
    "click",
    checkMessage
);


/* --------------------------------------------------
   Enter shortcut
-------------------------------------------------- */

messageInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {
            checkMessage();
        }

    }
);


/* --------------------------------------------------
   Main prediction function
-------------------------------------------------- */

async function checkMessage() {

    const message =
        messageInput.value.trim();


    // Validate input

    if (!message) {

        messageInput.focus();

        messageInput.style.borderColor =
            "rgba(255, 113, 137, 0.7)";

        setTimeout(() => {

            messageInput.style.borderColor =
                "";

        }, 1200);

        return;
    }


    // UI state

    checkButton.disabled = true;

    checkButton.style.opacity = "0.65";

    resultCard.classList.add("hidden");

    loading.classList.remove("hidden");


    try {

        /*
         * Send message to FastAPI
         */

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    message: message
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to analyze message."
            );

        }


        /*
         * Small artificial delay
         * makes the loading animation
         * feel smoother.
         */

        await delay(500);


        showResult(data);


    } catch (error) {

        showError(
            error.message
        );

    } finally {

        loading.classList.add("hidden");

        checkButton.disabled = false;

        checkButton.style.opacity = "1";

    }
}


/* --------------------------------------------------
   Display result
-------------------------------------------------- */

function showResult(data) {

    const isSpam =
        data.prediction === "SPAM";


    prediction.textContent =
        data.prediction;


    resultMessage.textContent =
        data.message;


    score.textContent =
        Number(data.decision_score)
            .toFixed(4);


    if (isSpam) {

        prediction.style.color =
            "var(--danger)";

        resultIcon.textContent =
            "!";

        resultIcon.style.color =
            "var(--danger)";

        resultIcon.style.background =
            "rgba(255, 113, 137, 0.12)";

        resultIcon.style.borderColor =
            "rgba(255, 113, 137, 0.22)";

    } else {

        prediction.style.color =
            "var(--success)";

        resultIcon.textContent =
            "✓";

        resultIcon.style.color =
            "var(--success)";

        resultIcon.style.background =
            "rgba(94, 230, 168, 0.13)";

        resultIcon.style.borderColor =
            "rgba(94, 230, 168, 0.22)";
    }


    /*
     * Convert SVM score into a visual
     * indicator only.
     *
     * This is NOT probability.
     */

    const absoluteScore =
        Math.min(
            Math.abs(
                Number(data.decision_score)
            ),
            2
        );

    const progress =
        Math.max(
            absoluteScore / 2 * 100,
            8
        );


    scoreProgress.style.width =
        "0%";


    resultCard.classList.remove(
        "hidden"
    );


    requestAnimationFrame(() => {

        scoreProgress.style.width =
            `${progress}%`;

    });
}


/* --------------------------------------------------
   Error state
-------------------------------------------------- */

function showError(message) {

    resultCard.classList.remove(
        "hidden"
    );

    prediction.textContent =
        "ERROR";

    prediction.style.color =
        "var(--danger)";

    resultIcon.textContent =
        "!";

    resultIcon.style.color =
        "var(--danger)";

    resultIcon.style.background =
        "rgba(255, 113, 137, 0.12)";

    resultMessage.textContent =
        message;

    score.textContent =
        "—";

    scoreProgress.style.width =
        "0%";
}


/* --------------------------------------------------
   Delay helper
-------------------------------------------------- */

function delay(milliseconds) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );
}