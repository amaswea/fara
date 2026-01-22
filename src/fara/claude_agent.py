from datetime import datetime
from typing import Any
from unittest import result
from playwright.async_api import Page
import logging
import json
import ast
import io
import os
from PIL import Image
from typing import List, Tuple, Dict
from urllib.parse import quote_plus
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from playwright.async_api import Download
from playwright.async_api import BrowserContext
import asyncio
from anthropic.types import TextBlock
from .browser.playwright_controller import PlaywrightController
from .fara_types import (
    LLMMessage,
    UserMessage,
    AssistantMessage,
    ImageObj,
    FunctionCall,
    message_to_anthropic_format,
    WebSurferEvent,
)
from anthropic import Anthropic

from .utils import get_trimmed_url


# TODO
# - Image filtering, use only 3 recent images
# - Add page text / dom
# - Ensure history / messages format
# - Implement more actions


class ClaudeAgent:
    DEFAULT_START_PAGE = "https://www.bing.com/"

    MLM_PROCESSOR_IM_CFG = {
        "min_pixels": 3136,
        "max_pixels": 12845056,
        "patch_size": 14,
        "merge_size": 2,
    }

    SCREENSHOT_TOKENS = 1105
    USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
    MAX_URL_LENGTH = 100
    MAX_TOKENS = 4096

    SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You control a Chromium browser via Playwright automation.
* The current date is {datetime.today().strftime("%A, %B %-d, %Y")}.
</SYSTEM_CAPABILITY>

<TOOL_GUIDANCE>
You receive a screenshot at the start of each turn. Look at it to see the current page - if you're already where you need to be, don't re-navigate.

After navigating to a new page, always call read_page to get element references (ref_1, ref_2, etc.) before interacting with the page. Use these refs with your interaction tools (click, type, hover, form_input, etc.). Refs are more reliable than coordinates.

When you need to extract or read text content from a page, always use get_page_text - don't try to read text from screenshots.

If DOM-based actions (refs) aren't working, fall back to screenshot + coordinate-based actions.
</TOOL_GUIDANCE>

<TIPS>
* Prefer get_page_text over scrolling when looking for information - it's faster and more reliable
* Use execute_js to extract data from JavaScript variables, localStorage, or trigger behaviors not accessible through clicks
* Use full URLs with https://
* Use wait for slow-loading pages
* Use scroll_to with a ref to reveal elements
* Use form_input with refs for form fields
* Use key for shortcuts (e.g., "ctrl+a")
* Close popups when they appear
* Verify actions succeeded before moving on
</TIPS>"""

    # Custom browser tool input schema
    BROWSER_TOOL_INPUT_SCHEMA: dict[str, Any] = {
        "properties": {
            "action": {
                "description": 'The action to perform. The available actions are:\n* `navigate`: Navigate to a URL or use "back"/"forward" for browser history navigation. Automatically includes a screenshot of the loaded page.\n* `screenshot`: Take a screenshot of the current browser viewport.\n* `left_click`: Click the left mouse button at the specified coordinate or element reference.\n* `right_click`: Click the right mouse button at the specified coordinate or element reference.\n* `middle_click`: Click the middle mouse button at the specified coordinate or element reference.\n* `double_click`: Double-click the left mouse button at the specified coordinate or element reference.\n* `triple_click`: Triple-click the left mouse button at the specified coordinate or element reference.\n* `hover`: Move the mouse cursor to the specified coordinate or element reference without clicking. Useful for revealing tooltips, dropdown menus, or triggering hover states.\n* `left_click_drag`: Click and drag from start_coordinate to coordinate.\n* `left_mouse_down`: Press and hold the left mouse button at the specified coordinate.\n* `left_mouse_up`: Release the left mouse button at the specified coordinate.\n* `scroll`: Scroll the page in a specified direction.\n* `scroll_to`: Scroll to bring an element into view.\n* `type`: Type text at the current cursor position.\n* `key`: Press a key or key combination (supports standard keys and modifiers).\n* `hold_key`: Hold down a key or key combination for a specified duration.\n* `read_page`: Get the DOM tree structure, optionally filtered for interactive elements.\n* `find`: Find text on the page and highlight matches.\n* `get_page_text`: Get all text content from the page.\n* `wait`: Wait for a specified duration in seconds.\n* `form_input`: Set the value of a form input element.\n* `zoom`: Take a zoomed screenshot of a specific region.\n* `execute_js`: Execute JavaScript code in the page context. Returns the result of the last expression.',
                "enum": [
                    "navigate",
                    "screenshot",
                    "left_click",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "triple_click",
                    "hover",
                    "left_click_drag",
                    "left_mouse_down",
                    "left_mouse_up",
                    "scroll",
                    "scroll_to",
                    "type",
                    "key",
                    "hold_key",
                    "read_page",
                    "find",
                    "get_page_text",
                    "wait",
                    "form_input",
                    "zoom",
                    "execute_js",
                ],
                "type": "string",
            },
            "text": {
                "description": 'Required for: `navigate` (URL or "back"/"forward"), `type` (text to type), `key` (key combination), `hold_key` (key to hold), `find` (text to search), `execute_js` (valid JavaScript code ONLY - no explanatory text, just the code). Optional for `read_page` (filter type: "interactive"), click actions (modifier keys to hold during click).',
                "type": "string",
            },
            "ref": {
                "description": "Element reference string for targeting specific DOM elements. Required for `scroll_to` and `form_input`. Optional for click actions and `hover` as an alternative to coordinates.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required for mouse actions when `ref` is not provided: `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click`, `hover`, `left_mouse_down`, `left_mouse_up`, `scroll`. Also serves as the end coordinate for `left_click_drag`.",
                "type": "array",
                "items": {"type": "integer"},
            },
            "start_coordinate": {
                "description": "(x, y): The starting x and y coordinates for drag operations. Required only for `left_click_drag`.",
                "type": "array",
                "items": {"type": "integer"},
            },
            "scroll_direction": {
                "description": "The direction to scroll. Required for `scroll` action.",
                "enum": ["up", "down", "left", "right"],
                "type": "string",
            },
            "scroll_amount": {
                "description": "The number of scroll units (similar to mouse wheel clicks). Required for `scroll` action.",
                "type": "integer",
            },
            "duration": {
                "description": "Duration in seconds. Required for `hold_key` and `wait` actions. For `wait`, must be between 0 and 100 seconds.",
                "type": "number",
            },
            "value": {
                "description": "The value to set for a form input element. Required for `form_input` action. Can be string, number, or boolean depending on the input type.",
                "type": ["string", "number", "boolean"],
            },
            "region": {
                "description": "(x1, y1, x2, y2): Defines a rectangular region for the `zoom` action. Coordinates specify top-left (x1, y1) and bottom-right (x2, y2) corners.",
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["action"],
        "type": "object",
    }

    BROWSER_TOOL_DESCRIPTION = """A browser automation tool for web interaction. Use this tool to navigate websites, interact with elements, and extract content.

    Key actions:
    - navigate: Go to a URL (automatically includes a screenshot)
    - screenshot: Take a visual screenshot
    - read_page: Get DOM structure with element references
    - get_page_text: Extract all text content
    - left_click, right_click, double_click: Click elements
    - hover: Move cursor without clicking (for tooltips, dropdowns)
    - type: Enter text at cursor
    - scroll: Scroll the page
    - form_input: Fill form fields
    - execute_js: Run JavaScript in page context"""

    def __init__(
        self,
        browser_manager: Any,
        client_config: dict,
        downloads_folder: str | None = None,
        start_page: str | None = "about:blank",
        animate_actions: bool = False,
        single_tab_mode: bool = True,
        max_n_images: int = 3,
        fn_call_template: str = "default",
        model_call_timeout: int = 20,
        max_rounds: int = 10,
        save_screenshots: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.downloads_folder = downloads_folder
        if not os.path.exists(self.downloads_folder or "") and self.downloads_folder:
            os.makedirs(self.downloads_folder)
        self.single_tab_mode = single_tab_mode
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.animate_actions = animate_actions
        self.browser_manager = browser_manager
        self.client_config = client_config
        self.max_n_images = max_n_images
        self.fn_call_template = fn_call_template
        self.model_call_timeout = model_call_timeout
        self.max_rounds = max_rounds
        self.max_url_chars = self.MAX_URL_LENGTH
        if save_screenshots and self.downloads_folder is None:
            assert False, "downloads_folder must be set if save_screenshots is True"
        self.save_screenshots = save_screenshots
        self._facts = []
        self._task_summary = None
        self._num_actions = 0
        self.logger = logger or logging.getLogger(__name__)
        self._mlm_width = 1440
        self._mlm_height = 900
        self.viewport_height = 900
        self.viewport_width = 1440
        self.include_input_text_key_args = True

        def _download_handler(download: Download) -> None:
            self._last_download = download

        self._download_handler = _download_handler
        self.did_initialize = False

        # Anthropic client will be initialized in initialize()
        self._anthropic_client: Anthropic | None = None
        self._chat_history: List[LLMMessage] = []

    async def initialize(self) -> None:
        if self.did_initialize:
            return
        self._last_download = None
        self._prior_metadata_hash = None

        # Initialize Anthropic client
        self._anthropic_client = Anthropic(api_key=self.client_config.get("api_key"))

        # Set up download handler
        self.browser_manager.set_download_handler(self._download_handler)

        # Initialize browser
        await self.browser_manager.init(self.start_page)
        self.did_initialize = True

    @property
    def _page(self) -> Page | None:
        """Get the current page from browser manager."""
        return self.browser_manager.page if self.browser_manager else None

    @_page.setter
    def _page(self, value):
        if self.browser_manager:
            self.browser_manager.page = value
        else:
            raise ValueError("Browser manager is not initialized. Cannot set page.")

    @property
    def context(self) -> BrowserContext | None:
        """Get the browser context from browser manager."""
        return self.browser_manager.context if self.browser_manager else None

    @property
    def _playwright_controller(self) -> PlaywrightController | None:
        """Get the playwright controller from browser manager."""
        return (
            self.browser_manager.playwright_controller if self.browser_manager else None
        )

    async def wait_for_captcha_with_timeout(
        self, timeout_seconds=300
    ):  # 5 minutes default
        """Wait for captcha to be solved with timeout"""
        try:
            await asyncio.wait_for(
                self.browser_manager.wait_for_captcha_resolution(),
                timeout=timeout_seconds,
            )
            return True  # Captcha solved in time
        except asyncio.TimeoutError:
            self.logger.warning(f"Captcha timeout after {timeout_seconds} seconds!")
            # Force resume execution
            self.browser_manager._captcha_event.set()
            return False  # Captcha timed out

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=5.0, min=5.0, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _make_model_call(
        self,
        history: List[LLMMessage],
        extra_create_args: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Make a model call using OpenAI client"""

        messages = [message_to_anthropic_format(msg) for msg in history]

        with open("claude_debug_messages.json", "w") as f:
            json.dump(messages, f, indent=2)

        tools = {
            "name": "browser",
            "description": self.BROWSER_TOOL_DESCRIPTION,
            "input_schema": self.BROWSER_TOOL_INPUT_SCHEMA,
        }

        api_kwargs = {
            "max_tokens": self.MAX_TOKENS,
            "messages": messages,
            "model": self.client_config.get("model", "claude-sonnet-4-5-20250929"),
            "system": self.SYSTEM_PROMPT,
            "tools": [tools],
            "temperature": 1.0,
        }

        # Use regular messages API when no beta features are needed
        try:
            response = self._anthropic_client.messages.create(**api_kwargs)
        except Exception as e:
            print("Error during model call:", e)
            self.logger.error(f"Error during model call: {e}", exc_info=True)
            raise e

        return response

    # def remove_screenshot_from_message(self, msg: List[Dict[str, Any]] | Any) -> Any:
    #     """Remove the screenshot from the message content."""
    #     if isinstance(msg.content, list):
    #         new_content = []
    #         for c in msg.content:
    #             if not isinstance(c, ImageObj):
    #                 new_content.append(c)
    #         msg.content = new_content
    #     elif isinstance(msg.content, ImageObj):
    #         msg = None
    #     return msg

    # def maybe_remove_old_screenshots(
    #     self, history: List[LLMMessage], includes_current: bool = False
    # ) -> List[LLMMessage]:
    #     """Remove old screenshots from the chat history. Assuming we have not yet added the current screenshot message.

    #     Note: Original user messages (marked with is_original=True) have their TEXT preserved,
    #     but their images may be removed if we exceed max_n_images. Boilerplate messages can be
    #     completely removed.
    #     """
    #     if self.max_n_images <= 0:
    #         return history

    #     max_n_images = self.max_n_images if includes_current else self.max_n_images - 1
    #     new_history: List[LLMMessage] = []
    #     n_images = 0
    #     for i in range(len(history) - 1, -1, -1):
    #         msg = history[i]

    #         is_original_user_message = isinstance(msg, UserMessage) and getattr(
    #             msg, "is_original", False
    #         )

    #         if i == 0 and n_images >= max_n_images:
    #             # First message is always the task so we keep it and remove the screenshot if necessary
    #             msg = self.remove_screenshot_from_message(msg)
    #             if msg is None:
    #                 continue

    #         if isinstance(msg.content, list):
    #             # Check if the message contains an image. Assumes 1 image per message.
    #             has_image = False
    #             for c in msg.content:
    #                 if isinstance(c, ImageObj):
    #                     has_image = True
    #                     break
    #             if has_image:
    #                 if n_images < max_n_images:
    #                     new_history.append(msg)
    #                 elif is_original_user_message:
    #                     # Original user message but over limit: keep text, remove image
    #                     msg = self.remove_screenshot_from_message(msg)
    #                     if msg is not None:
    #                         new_history.append(msg)
    #                 n_images += 1
    #             else:
    #                 new_history.append(msg)
    #         elif isinstance(msg.content, ImageObj):
    #             if n_images < max_n_images:
    #                 new_history.append(msg)
    #             n_images += 1
    #         else:
    #             new_history.append(msg)

    #     new_history = new_history[::-1]

    #     return new_history

    async def _get_screenshot(self) -> Image.Image:
        """Get current screenshot and scale it for the model."""
        screenshot = await self._playwright_controller.get_screenshot(self._page)
        screenshot = Image.open(io.BytesIO(screenshot))
        return screenshot

    # def _get_system_message(
    #     self, screenshot: ImageObj | Image.Image
    # ) -> Tuple[List[SystemMessage], Image.Image]:
    #     system_prompt_info = get_computer_use_system_prompt(
    #         screenshot,
    #         self.MLM_PROCESSOR_IM_CFG,
    #         include_input_text_key_args=self.include_input_text_key_args,
    #         fn_call_template=self.fn_call_template,
    #     )
    #     self._mlm_width, self._mlm_height = system_prompt_info["im_size"]
    #     scaled_screenshot = screenshot.resize((self._mlm_width, self._mlm_height))

    #     system_message = []
    #     for msg in system_prompt_info["conversation"]:
    #         tmp_content = ""
    #         for content in msg["content"]:
    #             tmp_content += content["text"]

    #         system_message.append(SystemMessage(content=tmp_content))

    #     return system_message, scaled_screenshot

    async def run(self, user_message: str) -> Tuple:
        """Run the agent with a user message."""
        # Initialize if not already done
        await self.initialize()

        # Ensure page is ready after initialization
        assert self._page is not None, "Page should be initialized"

        # Get initial screenshot and add user message with image to chat history
        screenshot = await self._get_screenshot()

        if self.save_screenshots:
            await self._playwright_controller.get_screenshot(
                self._page,
                path=os.path.join(
                    self.downloads_folder, f"screenshot{self._num_actions}.png"
                ),
            )

        # Initial user message does not have screenshot for claude.
        self._chat_history.append(
            UserMessage(
                content=[user_message],
                is_original=True,
            )
        )

        all_actions = []
        all_observations = []
        final_answer = "<no_answer>"
        prior_tool_use = None
        prior_tool_output = None
        for i in range(self.max_rounds):
            is_first_round = i == 0
            if not self.browser_manager._captcha_event.is_set():
                self.logger.info("Waiting 60s for captcha to finish...")
                captcha_solved = await self.wait_for_captcha_with_timeout(60)
                if (
                    not captcha_solved
                    and not self.browser_manager._captcha_event.is_set()
                ):
                    raise RuntimeError(
                        "Captcha timed out, unable to proceed with web surfing."
                    )

            tool_use, text_response = await self.generate_model_call(
                is_first_round,
                screenshot if is_first_round else None,
                prior_tool_result=prior_tool_use,
                prior_tool_output=prior_tool_output,
            )

            if text_response is not None:
                all_actions.append(text_response)

            prior_tool_use = tool_use
            if tool_use is not None:
                function_calls = [
                    FunctionCall(
                        id="dummy", name=tool_use.name, arguments=tool_use.input
                    )
                ]
                (
                    _,
                    _,
                    tool_output,
                ) = await self.execute_action(function_calls)
                prior_tool_output = tool_output

                all_observations.append(tool_output)
                self.logger.debug(f"Observation#{i+1}: {tool_output}")
                print(f"Observation#{i+1}: {tool_output}")

            if tool_use is None:
                final_answer = text_response
                break
        return final_answer, all_actions, all_observations

    async def generate_model_call(
        self,
        is_first_round: bool,
        first_screenshot: Image.Image | None = None,
        prior_tool_result: dict | None = None,
        prior_tool_output: dict | None = None,
    ) -> Tuple[List[FunctionCall], str]:
        # screenshot_for_system = first_screenshot
        if not is_first_round:
            # Get screenshot and add new user message for subsequent rounds
            screenshot = await self._get_screenshot()
            # screenshot_for_system = screenshot

            # text_prompt = self.USER_MESSAGE
            # curr_url = await self._playwright_controller.get_page_url(self._page)
            # trimmed_url = get_trimmed_url(curr_url, max_len=self.max_url_chars)
            # text_prompt = f"Current URL: {trimmed_url}\n" + text_prompt

            curr_message = UserMessage(
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": prior_tool_result.id,
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": ImageObj.from_pil(screenshot).to_base64(),
                                },
                            },
                            {"type": "text", "text": prior_tool_output},
                        ],
                    }
                ]
            )

            # curr_message = UserMessage(
            #     content=[ImageObj.from_pil(screenshot), text_prompt]
            # )
            self._chat_history.append(curr_message)

        # Generate system message using the screenshot

        # Don't add the FARA system prompt for anthropic
        # system_message, _ = self._get_system_message(screenshot_for_system)
        # history = system_message + history
        response = await self._make_model_call(
            self._chat_history, extra_create_args={"temperature": 0}
        )

        # Append assistant message to chat history
        text_response = next(
            (block.text for block in response.content if isinstance(block, TextBlock)),
            None,
        )

        tool_use = None
        if response.stop_reason == "tool_use":
            tool_use = next(
                block for block in response.content if block.type == "tool_use"
            )

        # Build the assistant message content
        assistant_content = []
        if text_response is not None:
            assistant_content.append({"type": "text", "text": text_response})
        if tool_use is not None:
            assistant_content.append(
                {
                    "type": tool_use.type,
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                }
            )
        self._chat_history.append(AssistantMessage(content=assistant_content))

        print("\n---Current Step ---")
        print("Tool:", tool_use)
        print("Text:", text_response)

        return tool_use, text_response

    async def execute_action(
        self,
        function_call: List[FunctionCall],
    ) -> Tuple[bool, bytes, str]:
        name = function_call[0].name
        args = function_call[0].arguments
        action_description = ""
        assert self._page is not None
        self.logger.debug(
            WebSurferEvent(
                source="ClaudeAgent",
                url=await self._playwright_controller.get_page_url(self._page),
                action=name,
                arguments=args,
                message=f"{name}( {json.dumps(args)} )",
            )
        )

        is_stop_action = False

        if args["action"] == "navigate":
            if "text" not in args:
                raise ValueError("navigate action requires 'text' argument")

            url = str(args["text"])
            action_description = f"Navigated to {url}"
            # Check if the argument starts with a known protocol
            if url.startswith(("https://", "http://", "file://", "about:")):
                (
                    reset_prior_metadata,
                    reset_last_download,
                ) = await self._playwright_controller.visit_page(self._page, url)
            # If the argument contains a space, treat it as a search query
            elif " " in url:
                (
                    reset_prior_metadata,
                    reset_last_download,
                ) = await self._playwright_controller.visit_page(
                    self._page,
                    f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH",
                )
            # Otherwise, prefix with https://
            else:
                (
                    reset_prior_metadata,
                    reset_last_download,
                ) = await self._playwright_controller.visit_page(
                    self._page, "https://" + url
                )
            if reset_last_download and self._last_download is not None:
                self._last_download = None
            if reset_prior_metadata and self._prior_metadata_hash is not None:
                self._prior_metadata_hash = None
        # elif args["action"] == "history_back":
        #     action_description = "I clicked the browser back button."
        #     await self._playwright_controller.back(self._page)
        # elif args["action"] == "web_search":
        #     query = args.get("query")
        #     action_description = f"I typed '{query}' into the browser search bar."
        #     encoded_query = quote_plus(query)
        #     (
        #         reset_prior_metadata,
        #         reset_last_download,
        #     ) = await self._playwright_controller.visit_page(
        #         self._page, f"https://www.bing.com/search?q={encoded_query}&FORM=QBLH"
        #     )
        #     if reset_last_download and self._last_download is not None:
        #         self._last_download = None
        #     if reset_prior_metadata and self._prior_metadata_hash is not None:
        #         self._prior_metadata_hash = None
        elif args["action"] == "scroll":
            scroll_direction = args.get("scroll_direction", "down")

            if scroll_direction == "up":
                action_description = "Scrolled up one page"
                self._playwright_controller.page_up()
            elif scroll_direction == "down":
                action_description = "Scrolled down one page"
                self._playwright_controller.page_down()

        # If other directions are needed, they can be implemented here

        elif args["action"] == "key":
            keys = args.get("text", [])
            action_description = f"Pressed key: {keys}"
            await self._playwright_controller.keypress(self._page, keys)
        elif args["action"] == "hover":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                await self._playwright_controller.hover_coords(self._page, tgt_x, tgt_y)

            action_description = f"Hovered at ({tgt_x}, {tgt_y})"

        elif args["action"] == "wait":
            duration = args.get("duration", 3.0)
            duration = args.get("time", duration)
            action_description = f"Waited for {duration} seconds."
            await self._playwright_controller.sleep(self._page, duration)
        elif (
            args["action"] == "click"
            or args["action"] == "left_click"
            or args["action"] == "double_click"
            or args["action"] == "triple_click"
            or args["action"] == "right_click"
            or args["action"] == "middle_click"
        ):
            if "coordinate" in args:
                button = "left"
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I clicked at coordinates ({tgt_x}, {tgt_y})."
                click_count = 1
                if args["action"] == "double_click":
                    click_count = 2
                elif args["action"] == "triple_click":
                    click_count = 3
                elif args["action"] == "right_click":
                    click_count = 1
                    button = "right"
                elif args["action"] == "middle_click":
                    click_count = 1
                    button = "middle"
                _ = await self._playwright_controller.click_coords(
                    self._page, tgt_x, tgt_y, button=button, count=click_count
                )

        elif args["action"] == "type":
            text_value = str(args.get("text", args.get("text_value")))
            action_description = f"Typed: '{text_value}'"
            await self._playwright_controller.type_text(self._page, text_value)
            # press_enter = args.get("press_enter", True)
            # delete_existing_text = args.get("delete_existing_text", False)

            # if "coordinate" in args:
            #     tgt_x, tgt_y = args["coordinate"]
            #     new_page_tentative = await self._playwright_controller.fill_coords(
            #         self._page,
            #         tgt_x,
            #         tgt_y,
            #         text_value,
            #         press_enter=press_enter,
            #         delete_existing_text=delete_existing_text,
            #     )
            #     if new_page_tentative is not None:
            #         self._page = new_page_tentative
            #         self._prior_metadata_hash = None

        # elif args["action"] == "pause_and_memorize_fact":
        #     fact = str(args.get("fact"))
        #     self._facts.append(fact)
        #     action_description = f"I memorized the following fact: {fact}"
        # elif args["action"] == "stop" or args["action"] == "terminate":
        #     action_description = args.get("thoughts")
        #     is_stop_action = True

        elif args["action"] == "screenshot":
            action_description = "Took a screenshot of the current page."

        elif args["action"] == "execute_js":
            js_code = str(args.get("text", ""))
            js_result = await self._playwright_controller.execute_script(
                self._page, js_code
            )
            action_description = f"Executed JavaScript code. Result: {js_result}"

        elif args["action"] == "read_page":
            dom_tree = await self._playwright_controller.get_page_dom(self._page)

            # The script returns {pageContent: string}, extract just the pageContent
            if isinstance(dom_tree, dict) and "pageContent" in dom_tree:
                full_content = dom_tree["pageContent"]
            elif isinstance(dom_tree, dict):
                full_content = json.dumps(dom_tree, indent=2)
            else:
                full_content = str(dom_tree)

            # Calculate content size for summary
            content_length = len(full_content)
            # Estimate token count
            # Note: For exact counts, use client.beta.messages.count_tokens API
            # This estimate uses ~3.5 chars/token which is typical for Claude with English text
            # Actual ratio varies by content type (code, languages, special characters)
            estimated_tokens = int(content_length / 3.5)

            # Create a summary for UI display
            url = result.get("url", "N/A") if isinstance(result, dict) else "N/A"
            summary = f"Extracted page DOM tree (~{estimated_tokens:,} tokens, {content_length:,} characters)"

            action_description = (
                f"__PAGE_EXTRACTED__\n{summary}\n__FULL_CONTENT__\n{full_content}"
            )
        elif args["action"] == "get_page_text":
            text = await self._playwright_controller.get_page_text(self._page)

            # Calculate content size for summary
            content_length = len(text)

            # Estimate token count
            # Note: For exact counts, use client.beta.messages.count_tokens API
            # This estimate uses ~3.5 chars/token which is typical for Claude with English text
            # Actual ratio varies by content type (code, languages, special characters)
            estimated_tokens = int(content_length / 3.5)

            # Create a summary for UI display
            url = result.get("url", "N/A") if isinstance(result, dict) else "N/A"
            summary = f"Extracted page text from \nURL: {url}\n(~{estimated_tokens:,} tokens, {content_length:,} characters)"

            action_description = (
                f"__PAGE_EXTRACTED__\n{summary}\n__FULL_CONTENT__\n{text}"
            )
        else:
            raise ValueError(f"Unknown tool: {args['action']}")

        await self._playwright_controller.wait_for_load_state(self._page)
        await self._playwright_controller.sleep(self._page, 3)

        # Get new screenshot after action
        self._num_actions += 1
        if self.save_screenshots:
            new_screenshot = await self._playwright_controller.get_screenshot(
                self._page,
                path=os.path.join(
                    self.downloads_folder, f"screenshot{self._num_actions}.png"
                ),
            )
        else:
            new_screenshot = await self._playwright_controller.get_screenshot(
                self._page
            )
        return is_stop_action, new_screenshot, action_description

    async def close(self) -> None:
        """
        Close the browser and the page.
        Should be called when the agent is no longer needed.
        """
        if self._page is not None:
            self._page = None
        await self.browser_manager.close()
