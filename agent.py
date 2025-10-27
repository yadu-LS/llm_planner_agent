import json
import os
import re
import calendar
import difflib
from dotenv import load_dotenv
import google.generativeai as genai
import urllib.request
from datetime import date, datetime

class PlannerAgent:
    VALID_CONSTRAINTS = ["Current", "Conservative", "Moderate", "Aggressive"]
    VALID_FORECAST_PERIODS = ["month", "2 month", "quarter", "6 month", "9 month", "year"]

    def __init__(self):
        self.plan_details = {
            "model_name": None,
            "forecast_period": None,
            "base_period": None,
            "constraint": None,
        }
        self.selected_model = None
        self.available_models = []
        self.chat_history = []
        self.called_functions = []
        
        try:
            load_dotenv()
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
            
            genai.configure(api_key=api_key)
            self.llm_service = genai.GenerativeModel('models/gemini-2.5-flash')
            print("Gemini API configured successfully.")

        except Exception as e:
            print(f"Error configuring the Gemini API: {e}")
            self.llm_service = None

    def no_op(self):
        pass

    def _fetch_models(self):
        bearer_token = os.environ.get("BEARER_TOKEN")
        if not bearer_token:
            return

        try:
            url = "https://console-platform-stg.lifesight.io/mmm/model?isArchived=true"
            headers = {"Authorization": f"Bearer {bearer_token}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    api_response = json.loads(response.read().decode())
                    self.available_models = api_response.get('data', [])
                    self.available_models = self.available_models[:10]
        except Exception as e:
            pass

    def fetch_and_select_model(self):
        if not self.available_models:
            return None
        return self.available_models

    def prompt_for_constraint(self):
        return self.VALID_CONSTRAINTS

    def validate_periods(self):
        forecast_period = self.plan_details.get('forecast_period')
        base_period = self.plan_details.get('base_period')

        if not forecast_period or not base_period:
            return False

        forecast_period_months = {
            "month": 1, "2 month": 2, "quarter": 3,
            "6 month": 6, "9 month": 9, "year": 12
        }
        
        forecast_months = forecast_period_months.get(forecast_period);
        if not forecast_months:
            return False

        try:
            base_start_str, base_end_str = base_period.split(' to ')
            base_start_date = datetime.strptime(base_start_str, '%Y-%m-%d')
            base_end_date = datetime.strptime(base_end_str, '%Y-%m-%d')
            
            # Calculate the number of months in the base period
            base_months = (base_end_date.year - base_start_date.year) * 12 + (base_end_date.month - base_start_date.month) + 1
            
            if forecast_months != base_months:
                return False
            return True
        except ValueError:
            return False

    def standardize_forecast_period(self, period_string):
        if not period_string or period_string in self.VALID_FORECAST_PERIODS:
            self.plan_details['forecast_period'] = period_string
            validation_result = self.validate_periods()
            self.chat_history.append({"role": "function", "name": "validate_periods", "content": str(validation_result)})
            return period_string

        today = date.today().strftime("%Y-%m-%d")

        llm_prompt = f"""
        Given the user's input for a forecast period and today's date, convert the input into one of the standard formats.

        Today's date: {today}
        User's input: "{period_string}"
        Standard formats: {', '.join(self.VALID_FORECAST_PERIODS)}

        Analyze the user's input. It could be a month name, a date range, or a relative duration. Calculate the duration of the period.
        - A single month (e.g., "january", "jan", "2025-01-01 to 2025-01-31") is "month".
        - A period of approximately 2 months is "2 month".
        - A period of approximately 3 months (a quarter, e.g., "2025-01-01 to 2025-03-31") is "quarter".
        - A period of approximately 6 months is "6 month".
        - A period of approximately 9 months is "9 month".
        - A period of approximately 12 months (a year) is "year".

        Return only the single, most appropriate standard format from the list.
        """

        try:
            response = self.llm_service.generate_content(llm_prompt)
            standardized_period = response.text.strip()

            if standardized_period in self.VALID_FORECAST_PERIODS:
                self.plan_details['forecast_period'] = standardized_period
                validation_result = self.validate_periods()
                self.chat_history.append({"role": "function", "name": "validate_periods", "content": str(validation_result)})
                return standardized_period
            else:
                self.plan_details['forecast_period'] = None # Invalidate it
                return None

        except Exception as e:
            return None

    def standardize_base_period(self, period_string):
        if not period_string:
            return None

        today = date.today().strftime("%Y-%m-%d")

        llm_prompt = f"""
        Given the user's input for a base period and today's date, convert the input into a specific date range format "yyyy-mm-dd to yyyy-mm-dd".

        Today's date: {today}
        User's input: "{period_string}"

        Analyze the user's input, which is a relative period. Calculate the start and end dates for this period.
        The start date should be the 1st of the starting month.
        The end date should be the last day of the ending month.

        For example:
        - If today is 2025-10-01 and the input is "last month", the output should be "2025-09-01 to 2025-09-30".
        - If today is 2025-10-01 and the input is "last quarter", the output should be "2025-07-01 to 2025-09-30".
        - If the input is "2024", the output should be "2024-01-01 to 2024-12-31".

        Return only the calculated date range in the format "yyyy-mm-dd to yyyy-mm-dd". Do not include any other text, reasoning, or explanation.
        """

        try:
            response = self.llm_service.generate_content(llm_prompt)
            standardized_period = response.text.strip()

            # Use regex to extract only the date range
            match = re.search(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", standardized_period)
            if match:
                self.plan_details['base_period'] = match.group(0)
                return match.group(0)
            else:
                self.plan_details['base_period'] = standardized_period # Fallback to the full response
                return standardized_period


        except Exception as e:
            self.plan_details['base_period'] = None # Invalidate it
            return None

    def _send_llm_request(self, user_prompt):
        if not self.llm_service:
            return "LLM service is not available."

        self.chat_history.append({"role": "user", "content": user_prompt})

        # Define the functions the LLM can call
        function_definitions = [
            {
                "name": "fetch_and_select_model",
                "description": "Fetches the available models and prompts the user to select one.",
                "parameters": []
            },
            {
                "name": "prompt_for_constraint",
                "description": "Prompts the user to select a constraint.",
                "parameters": []
            },
            {
                "name": "standardize_forecast_period",
                "description": "Standardizes the forecast period to a valid format.",
                "parameters": [
                    {"name": "period_string", "type": "string", "description": "The forecast period to standardize."}
                ]
            },
            {
                "name": "standardize_base_period",
                "description": "Standardizes the base period to a date range format.",
                "parameters": [
                    {"name": "period_string", "type": "string", "description": "The base period to standardize."}
                ]
            },
            {
                "name": "_calculate_default_base_period",
                "description": "Calculates the default base period.",
                "parameters": []
            },
            {
                "name": "validate_periods",
                "description": "Validates that the forecast and base periods have the same duration.",
                "parameters": []
            }
        ]

        # Construct the prompt
        full_prompt = f"""You are a marketing planner assistant. Your goal is to help the user create a marketing plan by gathering the necessary information.

        Here is the current state of the plan:
        {json.dumps(self.plan_details, indent=2)}

        Here is the conversation history:
        {json.dumps(self.chat_history, indent=2)}

        Here are the functions you can call:
        {json.dumps(function_definitions, indent=2)}

        Please analyze the user's latest prompt and the conversation history to understand their request. Based on this, you should either:
        1. Call one of the provided functions with the appropriate arguments.
        2. Ask for more information from the user in a friendly and conversational way. If you are asking for information, please list the missing information in your response.
        3. If all the information is gathered, confirm with the user and present a summary of the plan.

        Your response should be a JSON object with three keys:
        - "function_to_call": The name of the function to call, or null if you are asking a question.
        - "arguments": A JSON object with the arguments for the function, or null.
        - "response_to_user": A user-friendly, natural language response to the user.

        User's latest prompt: "{user_prompt}"
        """

        try:
            response = self.llm_service.generate_content(full_prompt)
            cleaned_json = response.text.strip().replace('```json', '').replace('```', '').strip()
            llm_response = json.loads(cleaned_json)

            response_to_user = llm_response.get("response_to_user", "I'm sorry, I didn't understand that. Could you please rephrase?")
            self.chat_history.append({"role": "agent", "content": response_to_user})

            function_to_call = llm_response.get("function_to_call")
            arguments = llm_response.get("arguments")

            if function_to_call:
                if hasattr(self, function_to_call):
                    if arguments:
                        getattr(self, function_to_call)(**arguments)
                    else:
                        getattr(self, function_to_call)()
                else:
                    print(f"Function {function_to_call} not found.")

            return response_to_user

        except Exception as e:
            print(f"An error occurred while processing the LLM response: {e}")
            return "I'm sorry, I encountered an error. Please try again."

    def _calculate_default_base_period(self):
        if not self.selected_model or not self.plan_details.get('forecast_period'):
            return None

        refresh_details = self.selected_model.get('refreshDetails')
        if not refresh_details:
            return None

        model_start_str = refresh_details[0].get('modelStart')
        if not model_start_str:
            return None

        try:
            model_start_date = date.fromisoformat(model_start_str)
        except (ValueError, TypeError):
            return None

        forecast_period_months = {
            "month": 1, "2 month": 2, "quarter": 3,
            "6 month": 6, "9 month": 9, "year": 12
        }
        
        forecast_months = forecast_period_months.get(self.plan_details['forecast_period'])
        if not forecast_months:
            return None

        # Calculate base_period_start_date
        start_month = model_start_date.month + 1
        start_year = model_start_date.year - 1
        if start_month > 12:
            start_month -= 12
            start_year += 1
        
        base_period_start_date = date(start_year, start_month, 1)

        # Calculate base_period_end_date
        end_month = base_period_start_date.month + forecast_months - 1
        end_year = base_period_start_date.year
        if end_month > 12:
            end_year += (end_month -1) // 12
            end_month = (end_month - 1) % 12 + 1

        _, end_day = calendar.monthrange(end_year, end_month)
        base_period_end_date = date(end_year, end_month, end_day)

        return f"{base_period_start_date.strftime('%Y-%m-%d')} to {base_period_end_date.strftime('%Y-%m-%d')}"

    def validate_periods(self):
        forecast_period = self.plan_details.get('forecast_period')
        base_period = self.plan_details.get('base_period')

        if not forecast_period or not base_period:
            return False

        forecast_period_months = {
            "month": 1, "2 month": 2, "quarter": 3,
            "6 month": 6, "9 month": 9, "year": 12
        }
        
        forecast_months = forecast_period_months.get(forecast_period)
        if not forecast_months:
            return False

        try:
            base_start_str, base_end_str = base_period.split(' to ')
            base_start_date = datetime.strptime(base_start_str, '%Y-%m-%d')
            base_end_date = datetime.strptime(base_end_str, '%Y-%m-%d')
            
            # Calculate the number of months in the base period
            base_months = (base_end_date.year - base_start_date.year) * 12 + (base_end_date.month - base_start_date.month) + 1
            
            if forecast_months != base_months:
                return False
            return True
        except ValueError:
            return False

    def no_op(self):
        pass

    def _send_llm_request(self, user_prompt):
        if not self.llm_service:
            return "LLM service is not available."

        self.chat_history.append({"role": "user", "content": user_prompt})

        # Define the functions the LLM can call
        function_definitions = [
            {
                "name": "fetch_and_select_model",
                "description": "Fetches the available models and prompts the user to select one. Call this function if the user provides a placeholder model name like 'test model'.",
                "parameters": []
            },
            {
                "name": "prompt_for_constraint",
                "description": "Prompts the user to select a constraint from the valid list.",
                "parameters": []
            },
            {
                "name": "standardize_forecast_period",
                "description": "Standardizes the forecast period to a valid format.",
                "parameters": [
                    {"name": "period_string", "type": "string", "description": "The forecast period to standardize."}
                ]
            },
            {
                "name": "standardize_base_period",
                "description": "Standardizes the base period to a date range format.",
                "parameters": [
                    {"name": "period_string", "type": "string", "description": "The base period to standardize."}
                ]
            },
            {
                "name": "_calculate_default_base_period",
                "description": "Calculates the default base period.",
                "parameters": []
            },
            {
                "name": "validate_periods",
                "description": "Validates that the forecast and base periods have the same duration.",
                "parameters": []
            },
            {
                "name": "no_op",
                "description": "Do nothing and just respond to the user.",
                "parameters": []
            }
        ]

        # Construct the prompt
        full_prompt = f"""You are a marketing planner assistant. Your goal is to help the user create a marketing plan by gathering the necessary information in a conversational way.

        Here is the current state of the plan:
        {json.dumps(self.plan_details, indent=2)}

        Here is the conversation history (including the results of function calls):
        {json.dumps(self.chat_history, indent=2)}

        Here are the functions you can call:
        {json.dumps(function_definitions, indent=2)}

        Please analyze the user's latest prompt and the entire conversation history to understand their request. Based on this, you should either:
        1. Call one of the provided functions with the appropriate arguments. Do not call a function if it has already been called successfully for the same purpose.
        2. Ask for more information from the user in a friendly and conversational way. Be direct and ask for all missing information at once. Do not ask for confirmation at every step.
        3. When asking for a constraint, you must present the user with the following options: {self.VALID_CONSTRAINTS}
        4. When asking for a forecast period, you can use the following as examples: {self.VALID_FORECAST_PERIODS}
        5. If the `validate_periods` function returns `False`, you must ask the user to update the base period to match the duration of the forecast period.
        6. If all the information is gathered (i.e., no values in `plan_details` are null), confirm with the user and present a detailed summary of the plan.

        Your response should be a JSON object with three keys:
        - "function_to_call": The name of the function to call, or "no_op" if you are just responding to the user.
        - "arguments": A JSON object with the arguments for the function, or null.
        - "response_to_user": A user-friendly, natural language response to the user.

        After a function is called, the result will be provided to you in the next turn with the role "function". Use this result to inform your next response and update the plan details.

        User's latest prompt: "{user_prompt}"
        """

        try:
            response = self.llm_service.generate_content(full_prompt)
            cleaned_json = response.text.strip().replace('```json', '').replace('```', '').strip()
            llm_response = json.loads(cleaned_json)

            response_to_user = llm_response.get("response_to_user", "I'm sorry, I didn't understand that. Could you please rephrase?")
            self.chat_history.append({"role": "agent", "content": response_to_user})

            function_to_call = llm_response.get("function_to_call")
            arguments = llm_response.get("arguments")

            if function_to_call and function_to_call != "no_op":
                self.called_functions.append(function_to_call)
                if hasattr(self, function_to_call):
                    if arguments:
                        function_result = getattr(self, function_to_call)(**arguments)
                    else:
                        function_result = getattr(self, function_to_call)()
                    
                    self.chat_history.append({"role": "function", "name": function_to_call, "content": str(function_result)})

                else:
                    print(f"Function {function_to_call} not found.")

            return response_to_user

        except Exception as e:
            print(f"An error occurred while processing the LLM response: {e}")
            return "I'm sorry, I encountered an error. Please try again."


    def run(self):
        if not self.llm_service:
            return

        self._fetch_models()

        print("\nHello! I am your marketing planner assistant. How can I help you today?")
        
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break

            response = self._send_llm_request(user_input)
            print(f"Agent: {response}")

if __name__ == '__main__':
    agent = PlannerAgent()
    agent.run()