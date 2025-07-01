import argparse
import json
import sys

from whatsapp import download_media as whatsapp_download_media
from whatsapp import get_chat as whatsapp_get_chat
from whatsapp import get_contact_chats as whatsapp_get_contact_chats
from whatsapp import get_direct_chat_by_contact as whatsapp_get_direct_chat_by_contact
from whatsapp import get_last_interaction as whatsapp_get_last_interaction
from whatsapp import get_message_context as whatsapp_get_message_context
from whatsapp import list_chats as whatsapp_list_chats
from whatsapp import list_messages as whatsapp_list_messages
from whatsapp import search_contacts as whatsapp_search_contacts
from whatsapp import send_audio_message as whatsapp_audio_voice_message
from whatsapp import send_file as whatsapp_send_file
from whatsapp import send_message as whatsapp_send_message


def main():
    parser = argparse.ArgumentParser(description="WhatsApp MCP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search_contacts
    sc = subparsers.add_parser(
        "search_contacts", help="Search WhatsApp contacts by name or phone number."
    )
    sc.add_argument(
        "query", help="Search term to match against contact names or phone numbers"
    )

    # list_messages
    lm = subparsers.add_parser(
        "list_messages", help="Get WhatsApp messages matching specified criteria."
    )
    lm.add_argument(
        "--after", help="Only return messages after this ISO-8601 date", default=None
    )
    lm.add_argument(
        "--before", help="Only return messages before this ISO-8601 date", default=None
    )
    lm.add_argument(
        "--sender_phone_number",
        help="Filter messages by sender phone number",
        default=None,
    )
    lm.add_argument("--chat_jid", help="Filter messages by chat JID", default=None)
    lm.add_argument(
        "--query", help="Search term to filter messages by content", default=None
    )
    lm.add_argument(
        "--limit", type=int, default=20, help="Maximum number of messages to return"
    )
    lm.add_argument("--page", type=int, default=0, help="Page number for pagination")
    lm.add_argument(
        "--include_context",
        action="store_true",
        help="Include messages before and after matches",
    )
    lm.add_argument(
        "--context_before",
        type=int,
        default=1,
        help="Messages to include before each match",
    )
    lm.add_argument(
        "--context_after",
        type=int,
        default=1,
        help="Messages to include after each match",
    )

    # list_chats
    lc = subparsers.add_parser(
        "list_chats", help="Get WhatsApp chats matching specified criteria."
    )
    lc.add_argument(
        "--query", help="Search term to filter chats by name or JID", default=None
    )
    lc.add_argument(
        "--limit", type=int, default=20, help="Maximum number of chats to return"
    )
    lc.add_argument("--page", type=int, default=0, help="Page number for pagination")
    lc.add_argument(
        "--include_last_message",
        action="store_true",
        help="Include the last message in each chat",
    )
    lc.add_argument(
        "--sort_by",
        choices=["last_active", "name"],
        default="last_active",
        help="Sort by field",
    )

    # get_chat
    gc = subparsers.add_parser("get_chat", help="Get WhatsApp chat metadata by JID.")
    gc.add_argument("chat_jid", help="The JID of the chat to retrieve")
    gc.add_argument(
        "--include_last_message", action="store_true", help="Include the last message"
    )

    # get_direct_chat_by_contact
    gdcbc = subparsers.add_parser(
        "get_direct_chat_by_contact",
        help="Get WhatsApp chat metadata by sender phone number.",
    )
    gdcbc.add_argument("sender_phone_number", help="The phone number to search for")

    # get_contact_chats
    gcc = subparsers.add_parser(
        "get_contact_chats", help="Get all WhatsApp chats involving the contact."
    )
    gcc.add_argument("jid", help="The contact's JID to search for")
    gcc.add_argument(
        "--limit", type=int, default=20, help="Maximum number of chats to return"
    )
    gcc.add_argument("--page", type=int, default=0, help="Page number for pagination")

    # get_last_interaction
    gli = subparsers.add_parser(
        "get_last_interaction",
        help="Get most recent WhatsApp message involving the contact.",
    )
    gli.add_argument("jid", help="The JID of the contact to search for")

    # get_message_context
    gmc = subparsers.add_parser(
        "get_message_context", help="Get context around a specific WhatsApp message."
    )
    gmc.add_argument("message_id", help="The ID of the message to get context for")
    gmc.add_argument(
        "--before",
        type=int,
        default=5,
        help="Messages to include before the target message",
    )
    gmc.add_argument(
        "--after",
        type=int,
        default=5,
        help="Messages to include after the target message",
    )

    # send_message
    sm = subparsers.add_parser(
        "send_message", help="Send a WhatsApp message to a person or group."
    )
    sm.add_argument("recipient", help="The recipient (phone number or JID)")
    sm.add_argument("message", help="The message text to send")

    # send_file
    sf = subparsers.add_parser(
        "send_file", help="Send a file via WhatsApp to the specified recipient."
    )
    sf.add_argument("recipient", help="The recipient (phone number or JID)")
    sf.add_argument("media_path", help="The absolute path to the media file to send")

    # send_audio_message
    sam = subparsers.add_parser(
        "send_audio_message", help="Send any audio file as a WhatsApp audio message."
    )
    sam.add_argument("recipient", help="The recipient (phone number or JID)")
    sam.add_argument("media_path", help="The absolute path to the audio file to send")

    # download_media
    dm = subparsers.add_parser(
        "download_media", help="Download media from a WhatsApp message."
    )
    dm.add_argument("message_id", help="The ID of the message containing the media")
    dm.add_argument("chat_jid", help="The JID of the chat containing the message")

    args = parser.parse_args()

    # Dispatch
    if args.command == "search_contacts":
        result = whatsapp_search_contacts(args.query)
    elif args.command == "list_messages":
        result = whatsapp_list_messages(
            after=args.after,
            before=args.before,
            sender_phone_number=args.sender_phone_number,
            chat_jid=args.chat_jid,
            query=args.query,
            limit=args.limit,
            page=args.page,
            include_context=args.include_context,
            context_before=args.context_before,
            context_after=args.context_after,
        )
    elif args.command == "list_chats":
        result = whatsapp_list_chats(
            query=args.query,
            limit=args.limit,
            page=args.page,
            include_last_message=args.include_last_message,
            sort_by=args.sort_by,
        )
    elif args.command == "get_chat":
        result = whatsapp_get_chat(args.chat_jid, args.include_last_message)
    elif args.command == "get_direct_chat_by_contact":
        result = whatsapp_get_direct_chat_by_contact(args.sender_phone_number)
    elif args.command == "get_contact_chats":
        result = whatsapp_get_contact_chats(args.jid, args.limit, args.page)
    elif args.command == "get_last_interaction":
        result = whatsapp_get_last_interaction(args.jid)
    elif args.command == "get_message_context":
        result = whatsapp_get_message_context(args.message_id, args.before, args.after)
    elif args.command == "send_message":
        success, status_message = whatsapp_send_message(args.recipient, args.message)
        result = {"success": success, "message": status_message}
    elif args.command == "send_file":
        success, status_message = whatsapp_send_file(args.recipient, args.media_path)
        result = {"success": success, "message": status_message}
    elif args.command == "send_audio_message":
        success, status_message = whatsapp_audio_voice_message(
            args.recipient, args.media_path
        )
        result = {"success": success, "message": status_message}
    elif args.command == "download_media":
        file_path = whatsapp_download_media(args.message_id, args.chat_jid)
        if file_path:
            result = {
                "success": True,
                "message": "Media downloaded successfully",
                "file_path": file_path,
            }
        else:
            result = {"success": False, "message": "Failed to download media"}
    else:
        parser.print_help()
        sys.exit(1)

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif hasattr(obj, "__dict__"):
            return to_serializable(vars(obj))
        elif hasattr(obj, "_asdict"):  # namedtuple
            return to_serializable(obj._asdict())
        else:
            return obj

    print(json.dumps(to_serializable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
