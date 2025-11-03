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
            "total_budget": None,
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

        except Exception as e:
            self.llm_service = None

    def no_op(self):
        pass

    def _fetch_models(self):
        bearer_token = os.environ.get("BEARER_TOKEN")
        if not bearer_token:
            print("DEBUG: BEARER_TOKEN is not set. Cannot fetch models.")
            return

        try:
            url = "https://console-platform-stg.lifesight.io/mmm/model?isArchived=true"
            headers = {"Authorization": f"Bearer {bearer_token}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                print(f"DEBUG: API Response Status: {response.status}")
                if response.status == 200:
                    api_response = json.loads(response.read().decode())
                    print(f"DEBUG: API Response: {json.dumps(api_response, indent=2)}")
                    self.available_models = api_response.get('data', [])
                    self.available_models = self.available_models[:10]
                    print(f"DEBUG: available_models: {self.available_models}") # Added debug print
                    if not self.available_models:
                        print("DEBUG: 'data' key in API response is empty or not found.")
                else:
                    print(f"DEBUG: Error fetching models: {response.status} - {response.reason}")
        except Exception as e:
            print(f"DEBUG: An error occurred while fetching models: {e}")

    def fetch_and_select_model(self):
        if not self.available_models:
            self._fetch_models()
        if not self.available_models:
            return "No models available."
        return [model['modelName'] for model in self.available_models]

    def select_model(self, model_name):
        if not self.available_models:
            self._fetch_models()

        try:
            model_names = [model['modelName'] for model in self.available_models]
        except KeyError:
            print("Error: The 'name' key was not found in one of the model objects.")
            print(f"Available models: {self.available_models}")
            return "An internal error occurred while selecting the model."

        if model_name in model_names:
            self.plan_details['model_name'] = model_name
            self.selected_model = next((model for model in self.available_models if model['modelName'] == model_name), None)
            return f"Model '{model_name}' has been selected."

        similar_models = difflib.get_close_matches(model_name, model_names, n=5)

        if similar_models:
            return f"Model '{model_name}' not found. Did you mean one of these? {', '.join(similar_models)}"
        else:
            return f"Model '{model_name}' not found. No similar models were found either. Please choose from the available models."

    def get_attribute_quality_score(self):
        if not self.selected_model:
            return "Please select a model first."

        sol_id = self.selected_model.get('solId')
        model_id = self.selected_model.get('id')

        if not sol_id or not model_id:
            return "Could not find solId or modelId for the selected model."

        bearer_token = os.environ.get("BEARER_TOKEN")
        if not bearer_token:
            print("DEBUG: BEARER_TOKEN is not set. Cannot fetch attribute quality score.")
            return "BEARER_TOKEN is not set. Cannot fetch attribute quality score."

        try:
            url = f"https://console-platform-stg.lifesight.io/mmm/model/attribute_quality_score?solId={sol_id}&modelId={model_id}"
            headers = {"Authorization": f"Bearer {bearer_token}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                print(f"DEBUG: Attribute Quality Score API Response Status: {response.status}")
                if response.status == 200:
                    api_response = json.loads(response.read().decode())
                    print(f"DEBUG: Attribute Quality Score API Response: {json.dumps(api_response, indent=2)}")
                    return api_response.get('data')
                else:
                    print(f"DEBUG: Error fetching attribute quality score: {response.status} - {response.reason}")
                    return f"Error fetching attribute quality score: {response.status} - {response.reason}"
        except Exception as e:
            print(f"DEBUG: An error occurred while fetching attribute quality score: {e}")
            return f"An error occurred while fetching attribute quality score: {e}"

    def get_channels_needing_calibration(self, quality_score_data):
        if not quality_score_data or 'channel_wise_attribute_info' not in quality_score_data:
            return []

        channels_needing_calibration = []
        channel_info = quality_score_data.get('channel_wise_attribute_info', {})

        for category in channel_info.values():
            if isinstance(category, list):
                for channel in category:
                    if isinstance(channel, dict) and channel.get('calibration') == 'YES':
                        channels_needing_calibration.append(channel.get('attribute'))

        return channels_needing_calibration

    def call_budget_optimizer(self, total_budget):
        if not self.selected_model:
            return "Please select a model first."
        if not self.plan_details.get('base_period'):
            return "Please set the base period first."
        if not self.plan_details.get('forecast_period'):
            return "Please set the forecast period first."

        self.plan_details['total_budget'] = total_budget

        model_id = self.selected_model.get('id')
        refresh_details = self.selected_model.get('refreshDetails')
        if not refresh_details or len(refresh_details) == 0:
            return "Refresh details not found in the selected model."
        
        input_data_start_date = refresh_details[0].get('modelStart')
        input_data_end_date = refresh_details[0].get('endDate')

        if not model_id or not input_data_start_date or not input_data_end_date:
            return "Model ID or input data dates are missing from the selected model."

        base_period = self.plan_details.get('base_period')
        try:
            start_date, end_date = base_period.split(' to ')
        except (ValueError, AttributeError):
            return "Invalid base period format. Expected 'YYYY-MM-DD to YYYY-MM-DD'."

        time_period = self.plan_details.get('forecast_period')

        payload = {
            "additiveModel": False,
            "budgetOptimiserScenario": "max_response",
            "endDate": end_date,
            "inputDataEndDate": input_data_end_date,
            "inputDataStartDate": input_data_start_date,
            "mmmRequestId": str(model_id),
            "startDate": start_date,
            "timePeriod": time_period,
            "totalBudget": total_budget
        }

        bearer_token = os.environ.get("BEARER_TOKEN")
        if not bearer_token:
            return "BEARER_TOKEN is not set."

        try:
            url = "https://console-platform-stg.lifesight.io/mmm/budgetOptimiser-default"
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    return f"Error calling budget optimizer: {response.status} - {response.reason}"
        except Exception as e:
            return f"An error occurred while calling budget optimizer: {e}"

    def _create_spend_table(self, budget_optimizer_result):
        if not budget_optimizer_result or not isinstance(budget_optimizer_result, dict):
            return "Invalid budget optimizer result."

        data = budget_optimizer_result.get('data')
        if not data or 'dateRangeToResponseMap' not in data:
            return "Could not find 'dateRangeToResponseMap' in budget optimizer response."

        date_range_map = data['dateRangeToResponseMap']
        if not date_range_map:
            return "No date range data found in budget optimizer response."

        first_date_range_key = next(iter(date_range_map))
        response_for_date_range = date_range_map[first_date_range_key]

        mmm_response_list = response_for_date_range.get('mmmBudgetOptimisationResponseList')
        if not mmm_response_list or not isinstance(mmm_response_list, list):
            return "Could not find 'mmmBudgetOptimisationResponseList' in budget optimizer response."

        constraint_data = {
            "Current": {"lower": 0.95, "upper": 1.05},
            "Conservative": {"lower": 0.75, "upper": 1.5},
            "Moderate": {"lower": 0.5, "upper": 2.0},
            "Aggressive": {"lower": 0.1, "upper": 4.99}
        }
        
        selected_constraint = self.plan_details.get('constraint')
        ratios = constraint_data.get(selected_constraint, {'lower': 1.0, 'upper': 1.0})
        lower_ratio = ratios['lower']
        upper_ratio = ratios['upper']

        table = "\n| Channel | Optimized Spend | Lower Limit | Upper Limit |\n|---|---|---|---|\n"
        for item in mmm_response_list:
            platform_name = item.get('platformName', 'N/A')
            optimized_spend = item.get('optimisedBudgetData', {}).get('spend', 0)
            
            lower_limit = optimized_spend * lower_ratio
            upper_limit = optimized_spend * upper_ratio

            table += f"| {platform_name} | {optimized_spend} | {lower_limit:.2f} | {upper_limit:.2f} |\n"
            
        return table

    def prompt_for_constraint(self):
        return self.VALID_CONSTRAINTS

    def set_constraint(self, constraint):
        if constraint in self.VALID_CONSTRAINTS:
            self.plan_details['constraint'] = constraint
            return f"Constraint has been set to '{constraint}'."
        else:
            return f"Invalid constraint: '{constraint}'. Please choose from {self.VALID_CONSTRAINTS}."

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

            match = re.search(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", standardized_period)
            if match:
                self.plan_details['base_period'] = match.group(0)
                return match.group(0)
            else:
                self.plan_details['base_period'] = standardized_period
                return standardized_period


        except Exception as e:
            self.plan_details['base_period'] = None
            return None

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

        start_month = model_start_date.month + 1
        start_year = model_start_date.year - 1
        if start_month > 12:
            start_month -= 12
            start_year += 1
        
        base_period_start_date = date(start_year, start_month, 1)

        end_month = base_period_start_date.month + forecast_months - 1
        end_year = base_period_start_date.year
        if end_month > 12:
            end_year += (end_month -1) // 12
            end_month = (end_month - 1) % 12 + 1

        _, end_day = calendar.monthrange(end_year, end_month)
        base_period_end_date = date(end_year, end_month, end_day)

        return f"{base_period_start_date.strftime('%Y-%m-%d')} to {base_period_end_date.strftime('%Y-%m-%d')}"

    def _send_llm_request(self, user_prompt):
        if not self.llm_service:
            return "LLM service is not available."

        self.chat_history.append({"role": "user", "content": user_prompt})

        function_definitions = [
            {
                "name": "fetch_and_select_model",
                "description": "Fetches the available models and prompts the user to select one. Call this function if the user provides a placeholder model name like 'test model'.",
                "parameters": []
            },
            {
                "name": "select_model",
                "description": "Selects a model and suggests similar models if the given model is not found.",
                "parameters": [
                    {"name": "model_name", "type": "string", "description": "The name of the model to select."}
                ]
            },
            {
                "name": "get_attribute_quality_score",
                "description": "Fetches the attribute quality score for the selected model to get information about channels to check the calibration of each channel.",
                "parameters": []
            },
            {
                "name": "call_budget_optimizer",
                "description": "Calls the budget optimizer to get channel-wise spend values. This should be called only after all other details of the plan are finalized.",
                "parameters": [
                    {"name": "total_budget", "type": "number", "description": "The total budget for the plan."}
                ]
            },
            {
                "name": "prompt_for_constraint",
                "description": "Prompts the user to select a constraint from the valid list.",
                "parameters": []
            },
            {
                "name": "set_constraint",
                "description": "Sets the constraint for the plan.",
                "parameters": [
                    {"name": "constraint", "type": "string", "description": "The constraint to set. Must be one of " + str(self.VALID_CONSTRAINTS)}
                ]
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

        full_prompt = f"""You are a marketing planner assistant. Your goal is to help the user create a marketing plan by gathering the necessary information in a conversational way.

        Here is the current state of the plan:
        {json.dumps(self.plan_details, indent=2)}

        Here is the conversation history (including the results of function calls):
        {json.dumps(self.chat_history, indent=2)}

        Here are the functions you can call:
        {json.dumps(function_definitions, indent=2)}

        Please analyze the user's latest prompt and the entire conversation history to understand their request. Based on this, you should either:
        1. **Proactive Information Gathering:** Your primary goal is to gather all necessary information (model, forecast period, base period, constraint, total budget). If the user has not provided all of this, ask for all missing pieces in a single, clear response. If the user provides all information at once, do not ask for step-by-step confirmation; proceed directly to the next logical step.
        2. Call one of the provided functions with the appropriate arguments. Do not call a function if it has already been called successfully for the same purpose.
        3. calculate the default value for baseperiod by calling _calculate_default_base_period function if baseperiod not specified by the user and ask the user if they want to keep it or change it if the want to chnage it tell them to enter the baseperiod and use that otherwie use the default value 
        4. When asking for a constraint, you must present the user with the following options: {self.VALID_CONSTRAINTS}
        5. When asking for a forecast period, you can use the following as examples: {self.VALID_FORECAST_PERIODS}
        6. If the `validate_periods` function returns `False`, you must ask the user to update the base period to match the duration of the forecast period.
        7. Once all information is gathered (model, periods, constraint, total_budget), confirm the details with the user. If the user confirms, call `call_budget_optimizer`.
        8. When a function result with the name `_create_spend_table` is available in the chat history, you MUST include the content of that result (which is a markdown table) as part of your `response_to_user`.

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

                    if function_to_call == 'get_attribute_quality_score' and isinstance(function_result, dict):
                        calibration_channels = self.get_channels_needing_calibration(function_result)
                        self.chat_history.append({"role": "function", "name": "get_channels_needing_calibration", "content": str(calibration_channels)})

                    if function_to_call == 'call_budget_optimizer' and isinstance(function_result, dict):
                        spend_table = self._create_spend_table(function_result)
                        self.chat_history.append({"role": "function", "name": "_create_spend_table", "content": spend_table})

                else:
                    print(f"Function {function_to_call} not found.")

            return response_to_user

        except Exception as e:
            print(f"An error occurred while processing the LLM response: {e}")
            return "I'm sorry, I encountered an error. Please try again."
