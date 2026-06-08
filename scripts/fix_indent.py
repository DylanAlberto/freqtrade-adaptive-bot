"""Fix indentation in strategy file."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

# Fix the unindented line
content = content.replace(
    '\n        """\np = self.',
    '\n        """\n        p = self.'
)

with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
    f.write(content)
print('Fixed')
