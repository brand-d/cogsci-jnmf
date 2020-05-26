import pandas as pd


# Load the raw data
raw_df = pd.read_csv('raw/dames2020_data.csv', delimiter=';')

# Extract syllog week 1
filter_columns = [
    'code',
    'syllog1_trial',
    'syllog',
    'syllog1_conclusion',
    'conscientiousness',
    'openness',
]

df = raw_df[filter_columns].rename(columns={
    'code': 'id',
    'syllog1_trial': 'sequence',
    'syllog': 'task_enc',
    'syllog1_conclusion': 'response_enc'
}).sort_values(['id', 'sequence']).reset_index(drop=True)

# Include CCOBRA task and response specification
def decode_task(enc_task):
    fig = int(enc_task[-1])
    q1 = enc_task[0].replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')
    q2 = enc_task[1].replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')

    if fig == 1:
        return '{};A;B/{};B;C'.format(q1, q2)
    elif fig == 2:
        return '{};B;A/{};C;B'.format(q1, q2)
    elif fig == 3:
        return '{};A;B/{};C;B'.format(q1, q2)
    elif fig == 4:
        return '{};B;A/{};B;C'.format(q1, q2)

    raise ValueError('Error: Invalid figure encountered.')

def decode_response(enc_resp):
    if enc_resp == 'nvc':
        return 'NVC'

    quant = enc_resp[0].upper().replace('E', 'No;').replace('A', 'All;').replace('I', 'Some;').replace('O', 'Some not;')
    direction = 'A;C' if enc_resp[1:] == 'ac' else 'C;A'
    return quant + direction

df['task'] = df['task_enc'].apply(decode_task)
df['response'] = df['response_enc'].apply(decode_response)

# Add choices
choices = []
for quant1 in ['A', 'I', 'E', 'O']:
    for direction in ['ac', 'ca']:
        choices.append(decode_response(quant1 + direction))
df['choices'] = '|'.join(choices)

# Add CCOBRA meta information
df['domain'] = 'syllogistic'
df['response_type'] = 'single-choice'

# Sort columns
df = df[[
    'id', 'sequence', 'task', 'choices', 'response', 'domain', 'response_type',
    'conscientiousness', 'openness'
]]

# Display and store
print(df.head())
df.to_csv('extracted.csv', index=False)
