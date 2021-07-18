# CheckersGamePlayingAIAgent

1. Developed a Checkers Game Playing AI Agent using Alpha Beta Pruning. The agent evaluated each position based on features such as having higher weight for protected pieces and advanced pieces, negative weight for vulnerable pieces etc.
2. The agent had to dynamically adjust an appropriate cut-off depth for computation to ensure that it does not time out the allowed 300 s game time limit for each side.
3. Grid Search was applied to identify the appropriate weights for evaluation metric features.
4. The agent was then run against a reference minimax agent with depth 3. It won 95% of the games and drew 5% of the remaining games.
