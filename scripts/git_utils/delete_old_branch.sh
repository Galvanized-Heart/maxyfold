# Ensure you're deleting the correct branch before running!
OLD_BRANCH_NAME="dataset"

git checkout main
git branch -d $OLD_BRANCH_NAME
git push origin --delete $OLD_BRANCH_NAME
git fetch --prune