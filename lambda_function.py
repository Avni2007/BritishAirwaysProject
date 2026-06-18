import json

def lambda_handler(event, context):

    review = event['review']

    response = {
        "sentiment": "Negative",
        "csat_score": 1,
        "topic": "Flight Delay",
        "recommendation": "Offer vouchers and improve scheduling"
    }

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }