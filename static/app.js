const thread = document.getElementById("thread");
const threadInner = document.getElementById("threadInner");

const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

const history = document.getElementById("history");
const newChat = document.getElementById("newChat");

let conversationStarted = false;


/* =========================
   SECURITY
   ========================= */

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}


/* =========================
   SCROLL
   ========================= */

function scrollToBottom() {
  requestAnimationFrame(() => {
    thread.scrollTo({
      top: thread.scrollHeight,
      behavior: "smooth"
    });
  });
}


/* =========================
   USER MESSAGE
   ========================= */

function addUserMessage(text) {

  const message = document.createElement("div");

  message.className = "message user";

  message.innerHTML = `
    <div class="message-label">
      You
    </div>

    <div class="user-bubble">
      ${escapeHtml(text)}
    </div>
  `;

  threadInner.appendChild(message);

  scrollToBottom();

  return message;
}


/* =========================
   THINKING STATE
   ========================= */

function addThinkingMessage() {

  const message = document.createElement("div");

  message.className = "message assistant";

  message.innerHTML = `
    <div class="message-label">
      GroundedRAG
    </div>

    <div class="thinking">

      <div class="thinking-orb"></div>

      <span>
        Searching indexed knowledge...
      </span>

    </div>
  `;

  threadInner.appendChild(message);

  scrollToBottom();

  return message;
}


/* =========================
   ASSISTANT RESPONSE
   ========================= */

function renderAssistant(result) {

  const message = document.createElement("div");

  message.className = "message assistant";

  const grounded = result.status === "ok";

  const badgeClass = grounded
    ? "grounded-badge"
    : "grounded-badge refused-badge";

  const status = grounded
    ? "Verified"
    : (result.status || "Refused");


  const answer = escapeHtml(
    result.answer || "No answer returned."
  ).replace(/\n/g, "<br>");


  let sources = "";


  if (result.sources && result.sources.length) {

    sources = `

      <div class="sources-wrapper">

        <div class="sources-title">

          <span>
            Retrieved evidence
          </span>

          <span>
            ${result.sources.length}
            source${result.sources.length === 1 ? "" : "s"}
          </span>

        </div>


        <div class="source-list">

          ${result.sources.map(source => {

            const score = Math.max(
              0,
              Math.min(1, Number(source.score || 0))
            );

            const percentage = Math.round(score * 100);

            const sourceId = escapeHtml(
              source.id || "Unknown source"
            );


            return `

              <div class="source">

                <div class="source-top">

                  <div class="source-left">

                    <div class="file-icon">
                      ◈
                    </div>

                    <div>

                      <div class="source-name">
                        ${sourceId}
                      </div>

                      <div class="source-chunk">
                        Retrieved document chunk
                      </div>

                    </div>

                  </div>

                  <div class="relevance">
                    ${percentage}%
                  </div>

                </div>


                <div class="relevance-bar">

                  <div
                    class="relevance-fill"
                    style="width:${percentage}%"
                  ></div>

                </div>

              </div>

            `;

          }).join("")}

        </div>

      </div>

    `;
  }


  message.innerHTML = `

    <div class="message-label">
      GroundedRAG
    </div>


    <div class="assistant-card">

      <div class="answer-card">

        <div class="answer-header">

          <span class="${badgeClass}">

            <span>
              ${grounded ? "●" : "!"}
            </span>

            ${escapeHtml(status)}

          </span>


          <span class="answer-tag">
            GROUNDED RESPONSE
          </span>

        </div>


        <div class="answer-content">
          ${answer}
        </div>


        <div class="answer-actions">

          <button
            class="small-btn copy-btn"
          >
            Copy
          </button>

        </div>


        ${sources}

      </div>

    </div>

  `;


  const copyButton =
    message.querySelector(".copy-btn");


  copyButton.addEventListener(
    "click",
    async () => {

      try {

        await navigator.clipboard.writeText(
          result.answer || ""
        );

        copyButton.textContent = "Copied";

        setTimeout(() => {
          copyButton.textContent = "Copy";
        }, 1200);

      } catch {

        copyButton.textContent = "Failed";

      }

    }
  );


  threadInner.appendChild(message);

  scrollToBottom();
}


/* =========================
   HISTORY
   ========================= */

function addHistoryItem(query) {

  const emptyMessage =
    history.querySelector(".history-empty");

  if (emptyMessage) {
    emptyMessage.remove();
  }


  const item = document.createElement("button");

  item.className = "history-item";

  item.textContent = query;

  item.title = query;


  item.addEventListener(
    "click",
    () => {

      input.value = query;

      input.focus();

    }
  );


  history.prepend(item);
}


/* =========================
   WELCOME
   ========================= */

function createWelcome() {

  const welcome = document.createElement("section");

  welcome.className = "welcome";

  welcome.id = "welcome";

  welcome.innerHTML = `

    <div class="welcome-orb">

      <span>✦</span>

    </div>


    <h1>
      Your knowledge.
      <span class="gradient-text">
        Grounded.
      </span>
    </h1>


    <p>
      Ask questions across your indexed documents.
      GroundedRAG retrieves relevant evidence before
      generating an answer, so unsupported claims
      don't get to sneak into the room.
    </p>


    <div class="suggestions">

      <button class="suggestion">

        <span class="suggestion-icon">
          ☁ CLOUD
        </span>

        <span class="suggestion-text">
          What is the default Cloud Run concurrency?
        </span>

      </button>


      <button class="suggestion">

        <span class="suggestion-icon">
          ⚡ CONFIG
        </span>

        <span class="suggestion-text">
          What are the key Cloud Run configuration options?
        </span>

      </button>


      <button class="suggestion">

        <span class="suggestion-icon">
          ◈ API
        </span>

        <span class="suggestion-text">
          How does the API gateway handle requests?
        </span>

      </button>


      <button class="suggestion">

        <span class="suggestion-icon">
          ◉ LIMITS
        </span>

        <span class="suggestion-text">
          What deployment limits should I know about?
        </span>

      </button>

    </div>

  `;


  attachSuggestionListeners(welcome);

  return welcome;
}


/* =========================
   SUGGESTIONS
   ========================= */

function attachSuggestionListeners(container) {

  container
    .querySelectorAll(".suggestion")
    .forEach(button => {

      button.addEventListener(
        "click",
        () => {

          const text =
            button
              .querySelector(".suggestion-text")
              ?.textContent
              .trim();

          if (!text) return;

          input.value = text;

          form.requestSubmit();

        }
      );

    });
}


/* =========================
   NEW CHAT
   ========================= */

newChat.addEventListener(
  "click",
  () => {

    threadInner.innerHTML = "";

    threadInner.appendChild(
      createWelcome()
    );

    conversationStarted = false;

    input.value = "";

    input.focus();

  }
);


/* =========================
   QUERY
   ========================= */

form.addEventListener(
  "submit",
  async event => {

    event.preventDefault();


    const query = input.value.trim();

    if (!query || sendBtn.disabled) {
      return;
    }


    if (!conversationStarted) {

      conversationStarted = true;

      const welcome =
        document.getElementById("welcome");

      if (welcome) {
        welcome.remove();
      }

    }


    addUserMessage(query);

    addHistoryItem(query);

    input.value = "";

    sendBtn.disabled = true;


    const thinking =
      addThinkingMessage();


    try {

      const response =
        await fetch("/query", {

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            query
          })

        });


      if (!response.ok) {

        throw new Error(
          `Server returned ${response.status}`
        );

      }


      const result =
        await response.json();


      thinking.remove();

      renderAssistant(result);


    } catch (error) {

      thinking.remove();


      renderAssistant({

        status: "error",

        answer:
          "The request could not be completed. " +
          error.message,

        sources: []

      });

    } finally {

      sendBtn.disabled = false;

      input.focus();

    }

  }
);


/* =========================
   KEYBOARD
   ========================= */

input.addEventListener(
  "keydown",
  event => {

    if (event.key === "Enter") {

      event.preventDefault();

      form.requestSubmit();

    }

  }
);
/* =========================
   SUGGESTED QUERY CARDS
   ========================= */

document.querySelectorAll("[data-query]").forEach(card => {
  card.addEventListener("click", () => {
    const query = card.dataset.query;

    if (!query) return;

    input.value = query;
    form.requestSubmit();
  });
});