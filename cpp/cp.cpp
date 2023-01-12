#include <bits/stdc++.h> 
using namespace std;

#define ll long long
#define mod 1000000007
#define INF (int)1e18

int abs(int a)
{
	if(a>0)
	{
		return a;
	}
	return -a;
}


void dfs(vector<list<int>> G, int node, int m, int a[])   // G graph, node-> starting node, m-> question specific, a[]->question specific
{
	int ans=0;
	int consecutive[G.size()];             

	for(int i=0; i<G.size(); i++) consecutive[i]=0;  //above two lines question specific

	bool visited[G.size()];                          //visited array
	for(int i=0; i<G.size(); i++) visited[i]=false;  //initializing visited array

	stack<pair<int,int>> s;                //usually stack<int> for dfs

	s.push(make_pair(node,a[node]));       //usually s.push(<int>) 

	while(!s.empty())
	{
		pair<int,int> curr = s.top();		 // usually int basically for popping the top
		s.pop();                             // popping the top element

		if(!visited[curr.first])
		{

			consecutive[curr.first] = curr.second;   
			//cout<<curr.first<<" ";                 // here the visiting step takes place.
			visited[curr.first] = true;
		}

		for(auto i: G[curr.first])                   //adding the non-visited neighbours to the stack
		{
			if(!visited[i])
			{	
				if(curr.second>=m+1)
				{
					s.push(make_pair(i,m+1));
				}
				else if(a[i]==0)
				{
					s.push(make_pair(i,0));
				}
				else 
				{
					s.push(make_pair(i,consecutive[curr.first]+1));	
				}
			}
		}
	}
}


// void dfs_recursive(vector<list<int>> G, int node, bool[] visited)
// {
// 	visited[node] = true;
// }

int main(){

	int n,m;
	cin>>n>>m;

	int a[n];

	for(int i=0; i<n; i++) cin>>a[i];

	vector<list<int>> G(n);
	
	for(int i=0; i<n-1; i++)
	{
		int a,b;
		cin>>a>>b;

		G[a-1].push_back(b-1);
		G[b-1].push_back(a-1);
	}

	dfs(G,0,m,a);

	return 0;
}
