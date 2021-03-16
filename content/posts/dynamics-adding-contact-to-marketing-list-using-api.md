---
title: "Dynamics Adding Contact to Marketing List Using Api"
date: 2021-03-14T18:48:46-04:00
showDate: true
draft: false
tags: ["blog","story"]
mermaid: false
---
Adding contact/lead/anything to a marketing list is very simple, however as with most Dynamics documentation isn't easy to find...

As an aside the key to interacting with Dynamics is the odata endpoint for your organisation that looks like:

[https://[organisationname].crm11.dynamics.com/api/data/v9.1/](https://[organisationname].dynamics.com/api/data/v9.1/) -replace organisation name with your org name

To add a contact to a marketing list all you have to do is submit a POST to the endpoint:

[https://[organisaction name].crm11.dynamics.com/api/data/v9.1/AddListMembersList](https://[organisaction%20name].crm11.dynamics.com/api/data/v9.1/AddListMembersList)

with the payload:

```
{ 
"List": { 
      "listid": "[ListID - of the marketing list]", 
      "@odata.type": "Microsoft.Dynamics.CRM.list" 
}, 
"Members": [ 
      { 
      "contactid": "[ContactID]", 
      "@odata.type": "Microsoft.Dynamics.CRM.contact" 
      } 
   ] 
}
```

Simple as that.

#### How do i get the ListID??

GET - [https://[organisaction name].crm11.dynamics.com/api/data/v9.1/lists](https://[organisaction%20name].crm11.dynamics.com/api/data/v9.0/lists)

This will return a list of all your lists, to make it readable use

GET - [https://[organisaction name].crm11.dynamics.com/api/data/v9.1/lists?$select=listname,listid](https://[organisaction%20name].crm11.dynamics.com/api/data/v9.0/lists?$select=listname,listid)

#### How do i get ContactID??

GET - [https://[organisation name].crm11.dynamics.com/api/data/v9.1/contacts?$select=fullname,contactid&$filter=lastname eq 'searchname'](https://[organisation%20name].crm11.dynamics.com/api/data/v9.1/contacts?$select=fullname,contactid&$filter=lastname%20eq%20%27searchname%27)

Will return all contacts with the lastname 'searchname'