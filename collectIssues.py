from github import Github
from github import Label
import json

# using an access token
g = Github("6085d91cefe7a650d320b2caad2f7eb2fabe82f0")

bugIssues = []
featureIssues = []
with open('baseRepos.json') as baseReposFile:
    baseRepos = json.load(baseReposFile)
    for language, repoNames in baseRepos.items():
        for repoAndLabelIssuesName in repoNames:
            repoName = list(repoAndLabelIssuesName.keys())[0]
            repo = g.get_repo(repoName)
            isBugIssues = True
            for labelName in list(repoAndLabelIssuesName.values())[0]:
                label = repo.get_label(labelName)
                issues = repo.get_issues(labels=[label])
                for issue in issues:
                    issueDict = issue.title + '. ' + issue.body
                    if isBugIssues:
                        bugIssues.append(issueDict)
                    else:
                        featureIssues.append(issueDict)
                isBugIssues = not isBugIssues
    
with open('baseIssues.json', 'w') as baseIssues:
    json.dump({'bug': bugIssues, 'feature': featureIssues}, baseIssues)
    print(len(bugIssues))
    print(len(featureIssues))
